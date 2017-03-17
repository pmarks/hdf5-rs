use ffi::h5i::{H5I_type_t, H5Iget_ref, hid_t};

use error::Result;
use handle::{Handle, get_id_type};

use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AllowTypes {
    Any,
    Just(H5I_type_t),
    OneOf(&'static [H5I_type_t]),
}

pub trait ObjectType : Sized {
    fn allow_types() -> AllowTypes;
    fn from_id(id: hid_t) -> Result<Self>;
    fn type_name() -> &'static str;

    fn describe(_: &Object<Self>) -> String {
        "".to_owned()
    }
}

impl<T: ObjectType> fmt::Debug for Object<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let out = if !self.is_valid() {
            format!("<HDF5 {}: invalid id>", T::type_name())
        } else {
            let desc = T::describe(self);
            if desc.is_empty() {
                format!("<HDF5 {}>", T::type_name())
            } else {
                format!("<HDF5 {}: {}>", T::type_name(), desc)
            }
        };
        fmt::Display::fmt(&out, f)
    }
}

impl ObjectType for () {
    fn allow_types() -> AllowTypes {
        AllowTypes::Any
    }

    fn from_id(_: hid_t) -> Result<()> {
        Ok(())
    }

    fn type_name() -> &'static str {
        "object"
    }
}

/// Any HDF5 object that can be referenced through an identifier.
pub struct Object<T: ObjectType> {
    handle: Handle,
    detail: T,
}

// TODO: this can be removed when feature(pub_restricted) lands
pub trait ObjectDetail<T: ObjectType> {
    fn detail(&self) -> &T;
}

impl<T: ObjectType> ObjectDetail<T> for Object<T> {
    fn detail(&self) -> &T {
        &self.detail
    }
}

// This internal trait provides raw access to the object handle.
pub trait ObjectID : Sized {
    fn id(&self) -> hid_t;
    fn from_id(id: hid_t) -> Result<Self>;
    fn incref(&self);
    fn decref(&self);
}

impl<T: ObjectType> ObjectID for Object<T> {
    fn id(&self) -> hid_t {
        self.handle.id()
    }

    fn from_id(id: hid_t) -> Result<Object<T>> {
        let allow_types = T::allow_types();
        if let AllowTypes::Just(cls_id) = allow_types {
            let id_type = get_id_type(id);
            ensure!(id_type == cls_id,
                    "Invalid {} id type: expected {:?}, got {:?}",
                    T::type_name(), cls_id, id_type);
        } else if let AllowTypes::OneOf(cls_ids) = allow_types {
            let id_type = get_id_type(id);
            ensure!(cls_ids.iter().find(|c| *c == &id_type).is_some(),
                    "Invalid {} id type: expected one of {:?}, got {:?}",
                    T::type_name(), cls_ids, id_type);
        }
        h5lock!({
            let detail = T::from_id(id)?;
            let handle = Handle::new(id)?;
            Ok(Object { handle: handle, detail: detail })
        })
    }

    fn incref(&self) {
        self.handle.incref();
    }

    fn decref(&self) {
        self.handle.decref();
    }
}

impl<T: ObjectType> Object<T> {
    /// Returns reference count if the handle is valid and 0 otherwise.
    pub fn refcount(&self) -> u32 {
        if self.is_valid() {
            match h5call!(H5Iget_ref(self.id())) {
                Ok(count) if count >= 0 => count as u32,
                _ => 0,
            }
        } else {
            0
        }
    }

    /// Returns `true` if the object has a valid unlocked identifier (`false` for pre-defined
    /// locked identifiers like property list classes).
    pub fn is_valid(&self) -> bool {
        self.handle.is_valid()
    }

    /// Returns type of the object.
    pub fn id_type(&self) -> H5I_type_t {
        get_id_type(self.id())
    }
}

#[cfg(test)]
pub mod tests {
    use ffi::h5i::{H5I_INVALID_HID, hid_t};
    use ffi::h5p::{H5P_DEFAULT, H5Pcreate};
    use globals::H5P_FILE_ACCESS;

    use super::{Object, ObjectType, AllowTypes, ObjectID};
    use error::Result;
    use handle::{is_valid_id, is_valid_user_id};

    struct TestObjectID;

    impl ObjectType for TestObjectID {
        fn allow_types() -> AllowTypes {
            AllowTypes::Any
        }

        fn from_id(_: hid_t) -> Result<TestObjectID> {
            Ok(TestObjectID)
        }

        fn type_name() -> &'static str {
            "test object"
        }

        fn describe(_: &TestObject) -> String {
            "foo".to_owned()
        }
    }

    type TestObject = Object<TestObjectID>;

    impl TestObject {
        fn incref(&self) {
            self.handle.incref()
        }

        fn decref(&self) {
            self.handle.decref()
        }
    }

    #[test]
    pub fn test_debug() {
        let obj = TestObject::from_id(
            h5call!(H5Pcreate(*H5P_FILE_ACCESS)).unwrap()).unwrap();
        assert_eq!(format!("{:?}", obj), "<HDF5 test object: foo>");
    }

    #[test]
    pub fn test_not_a_valid_user_id() {
        assert_err!(TestObject::from_id(H5I_INVALID_HID), "Invalid handle id");
        assert_err!(TestObject::from_id(H5P_DEFAULT), "Invalid handle id");
    }

    #[test]
    pub fn test_new_user_id() {
        let obj = TestObject::from_id(
            h5call!(H5Pcreate(*H5P_FILE_ACCESS)).unwrap()).unwrap();
        assert!(obj.is_valid());
        assert!(obj.id() > 0);
        assert!(is_valid_id(obj.id()));
        assert!(is_valid_user_id(obj.id()));

        assert_eq!(obj.refcount(), 1);
        obj.incref();
        assert_eq!(obj.refcount(), 2);
        obj.decref();
        assert_eq!(obj.refcount(), 1);
        obj.decref();
        obj.decref();
        assert_eq!(obj.refcount(), 0);
        assert!(!obj.is_valid());
        assert!(!is_valid_user_id(obj.id()));
        assert!(!is_valid_id(obj.id()));
    }

    #[test]
    pub fn test_incref_decref_drop() {
        let mut obj = TestObject::from_id(
            h5call!(H5Pcreate(*H5P_FILE_ACCESS)).unwrap()).unwrap();
        let obj_id = obj.id();
        obj = TestObject::from_id(h5call!(H5Pcreate(*H5P_FILE_ACCESS)).unwrap()).unwrap();
        assert_ne!(obj_id, obj.id());
        assert!(obj.id() > 0);
        assert!(obj.is_valid());
        assert!(is_valid_id(obj.id()));
        assert!(is_valid_user_id(obj.id()));
        assert_eq!(obj.refcount(), 1);
        let mut obj2 = TestObject::from_id(obj.id()).unwrap();
        obj2.incref();
        assert_eq!(obj.refcount(), 2);
        assert_eq!(obj2.refcount(), 2);
        drop(obj2);
        assert!(obj.is_valid());
        assert_eq!(obj.refcount(), 1);
        obj2 = TestObject::from_id(obj.id()).unwrap();
        obj2.incref();
        obj.decref();
        obj.decref();
        assert_eq!(obj.id(), H5I_INVALID_HID);
        assert_eq!(obj2.id(), H5I_INVALID_HID);
    }
}
