use error::Result;
use object::{Object, ObjectType, AllowTypes, ObjectID};

use ffi::h5i::{H5I_DATATYPE, hid_t};
use ffi::h5t::{
    H5T_INTEGER, H5T_FLOAT, H5T_NO_CLASS, H5T_NCLASSES, H5T_ORDER_BE, H5T_ORDER_LE, H5T_SGN_2,
    H5Tcopy, H5Tget_class, H5Tget_order, H5Tget_offset, H5Tget_sign, H5Tget_precision, H5Tget_size,
    H5Tequal
};

use libc::c_void;
use std::fmt;
use std::mem;

#[cfg(target_endian = "big")]
use globals::{
    H5T_STD_I8BE, H5T_STD_I16BE,
    H5T_STD_I32BE, H5T_STD_I64BE,
    H5T_STD_U8BE, H5T_STD_U16BE,
    H5T_STD_U32BE, H5T_STD_U64BE,
    H5T_IEEE_F32BE, H5T_IEEE_F64BE,
};

#[cfg(target_endian = "little")]
use globals::{
    H5T_STD_I8LE, H5T_STD_I16LE,
    H5T_STD_I32LE, H5T_STD_I64LE,
    H5T_STD_U8LE, H5T_STD_U16LE,
    H5T_STD_U32LE, H5T_STD_U64LE,
    H5T_IEEE_F32LE, H5T_IEEE_F64LE,
};

/// A trait for all HDF5 datatypes.
pub trait AnyDatatype : ObjectType {}

impl<T: AnyDatatype> Object<T> {
    /// Get the total size of the datatype in bytes.
    pub fn size(&self) -> usize {
        h5call!(H5Tget_size(self.id())).unwrap_or(0) as usize
    }
}

macro_rules! def_atomic {
    ($name:ident -> $alias:ident, $h5t:ident, $desc:expr) => (
        pub struct $name;

        impl ObjectType for $name {
            fn allow_types() -> AllowTypes {
                AllowTypes::Just(H5I_DATATYPE)
            }

            fn from_id(id: hid_t) -> Result<$name> {
                h5lock!({
                    let cls = H5Tget_class(id);
                    ensure!(cls == $h5t, "Invalid datatype class: expected {:?}, got {:?}",
                            $h5t, cls);
                    Ok($name)
                })
            }

            fn type_name() -> &'static str {
                $desc
            }
        }

        impl AnyDatatype for $name {}
        impl AtomicDatatype for $name {}

        pub type $alias = Object<$name>;
    )
}

/// A trait for integer scalar datatypes.
def_atomic!(IntegerDatatypeID -> IntegerDatatype, H5T_INTEGER, "integer datatype");

impl IntegerDatatype {
    /// Returns true if the datatype is signed.
    pub fn is_signed(&self) -> bool {
        h5lock!(H5Tget_sign(self.id()) == H5T_SGN_2)
    }
}

/// A trait for floating-point scalar datatypes.
def_atomic!(FloatDatatypeID -> FloatDatatype, H5T_FLOAT, "float datatype");

/// A trait for atomic scalar datatypes.
pub trait AtomicDatatype : AnyDatatype {}

impl<T: AtomicDatatype> Object<T> {
    /// Returns true if the datatype byte order is big endian.
    pub fn is_be(&self) -> bool {
        h5lock!(H5Tget_order(self.id()) == H5T_ORDER_BE)
    }

    /// Returns true if the datatype byte order is little endian.
    pub fn is_le(&self) -> bool {
        h5lock!(H5Tget_order(self.id()) == H5T_ORDER_LE)
    }

    /// Get the offset of the first significant bit.
    pub fn offset(&self) -> usize {
        h5call!(H5Tget_offset(self.id())).unwrap_or(0) as usize
    }

    /// Get the number of significant bits, excluding padding.
    pub fn precision(&self) -> usize {
        h5call!(H5Tget_precision(self.id())).unwrap_or(0) as usize
    }
}

/// A trait for native types that are convertible to HDF5 datatypes.
pub trait ToDatatype: Clone {
    fn to_datatype() -> Result<Datatype>;
    fn from_raw_ptr(buf: *const c_void) -> Self;
    fn with_raw_ptr<T, F: Fn(*const c_void) -> T>(value: Self, func: F) -> T;
}

macro_rules! impl_atomic {
    ($tp:ty, $be:ident, $le:ident) => (
        impl ToDatatype for $tp {
            #[cfg(target_endian = "big")]
            fn to_datatype() -> Result<Datatype> {
                Datatype::from_id(h5try!(H5Tcopy(*$be)))
            }

            #[cfg(target_endian = "little")]
            fn to_datatype() -> Result<Datatype> {
                Datatype::from_id(h5try!(H5Tcopy(*$le)))
            }

            fn with_raw_ptr<T, F: Fn(*const c_void) -> T>(value: Self, func: F) -> T {
                let buf = &value as *const _ as *const c_void;
                func(buf)
            }

            fn from_raw_ptr(buf: *const c_void) -> Self {
                unsafe { *(buf as *const Self) }
            }
        }
    )
}

impl_atomic!(bool, H5T_STD_U8BE, H5T_STD_U8LE);

impl_atomic!(i8, H5T_STD_I8BE, H5T_STD_I8LE);
impl_atomic!(i16, H5T_STD_I16BE, H5T_STD_I16LE);
impl_atomic!(i32, H5T_STD_I32BE, H5T_STD_I32LE);
impl_atomic!(i64, H5T_STD_I64BE, H5T_STD_I64LE);

impl_atomic!(u8, H5T_STD_U8BE, H5T_STD_U8LE);
impl_atomic!(u16, H5T_STD_U16BE, H5T_STD_U16LE);
impl_atomic!(u32, H5T_STD_U32BE, H5T_STD_U32LE);
impl_atomic!(u64, H5T_STD_U64BE, H5T_STD_U64LE);

impl_atomic!(f32, H5T_IEEE_F32BE, H5T_IEEE_F32LE);
impl_atomic!(f64, H5T_IEEE_F64BE, H5T_IEEE_F64LE);

#[cfg(target_pointer_width = "32")] impl_atomic!(usize, H5T_STD_U32BE, H5T_STD_U32LE);
#[cfg(target_pointer_width = "32")] impl_atomic!(isize, H5T_STD_I32BE, H5T_STD_I32LE);

#[cfg(target_pointer_width = "64")] impl_atomic!(usize, H5T_STD_U64BE, H5T_STD_U64LE);
#[cfg(target_pointer_width = "64")] impl_atomic!(isize, H5T_STD_I64BE, H5T_STD_I64LE);

pub enum DatatypeID {
    Integer,
    Float,
}

/// Represents the HDF5 datatype object.
pub type Datatype = Object<DatatypeID>;

impl ObjectType for DatatypeID {
    fn allow_types() -> AllowTypes {
        AllowTypes::Just(H5I_DATATYPE)
    }

    fn from_id(id: hid_t) -> Result<DatatypeID> {
        h5lock!({
            match H5Tget_class(id) {
                H5T_INTEGER  => Ok(DatatypeID::Integer),
                H5T_FLOAT    => Ok(DatatypeID::Float),
                H5T_NO_CLASS |
                H5T_NCLASSES => Err(From::from("Invalid datatype class")),
                cls          => Err(From::from(format!("Unsupported datatype: {:?}", cls))),
            }
        })
    }

    fn type_name() -> &'static str {
        "datatype"
    }
}

pub enum DatatypeClass<'a> {
    Integer(&'a IntegerDatatype),
    Float(&'a FloatDatatype),
}

impl Datatype {
    pub fn class<'a>(&'a self) -> Result<DatatypeClass<'a>> {
        h5lock!({
            match H5Tget_class(self.id()) {
                H5T_INTEGER  => Ok(DatatypeClass::Integer(mem::transmute(self))),
                H5T_FLOAT    => Ok(DatatypeClass::Float(mem::transmute(self))),
                H5T_NO_CLASS |
                H5T_NCLASSES => Err(From::from("Invalid datatype class")),
                cls          => Err(From::from(format!("Unsupported datatype: {:?}", cls))),
            }
        })
    }
}

impl AnyDatatype for DatatypeID {}

impl PartialEq for Datatype {
    fn eq(&self, other: &Datatype) -> bool {
        h5call!(H5Tequal(self.id(), other.id())).unwrap_or(0) == 1
    }
}

impl fmt::Debug for IntegerDatatype {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for IntegerDatatype {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if !self.is_valid() {
            return "<HDF5 datatype: invalid id>".fmt(f);
        }
        format!("<HDF5 datatype: {}-bit {}signed integer>",
                self.precision(), if self.is_signed() { "" } else { "un" }).fmt(f)
    }
}

impl fmt::Debug for FloatDatatype {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for FloatDatatype {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if !self.is_valid() {
            return "<HDF5 datatype: invalid id>".fmt(f);
        }
        format!("<HDF5 datatype: {}-bit float>", self.precision()).fmt(f)
    }
}

impl fmt::Debug for Datatype {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for Datatype {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if !self.is_valid() {
            return "<HDF5 datatype: invalid id>".fmt(f);
        }
        match self.class() {
            Ok(dt) => match dt {
                DatatypeClass::Integer(dt) => dt.fmt(f),
                DatatypeClass::Float(dt) => dt.fmt(f),
            },
            Err(_) => "<HDF5 datatype: invalid class>".fmt(f),
        }
    }
}

#[cfg(test)]
pub mod tests {
    use super::{Datatype, DatatypeClass, ToDatatype};
    use ffi::h5i::H5I_INVALID_HID;
    use ffi::h5t::H5Tcopy;
    use globals::H5T_STD_REF_OBJ;
    use object::ObjectID;

    #[cfg(target_endian = "big")] const IS_BE: bool = true;
    #[cfg(target_endian = "big")] const IS_LE: bool = false;

    #[cfg(target_endian = "little")] const IS_BE: bool = false;
    #[cfg(target_endian = "little")] const IS_LE: bool = true;

    #[cfg(target_pointer_width = "32")] const POINTER_WIDTH_BYTES: usize = 4;
    #[cfg(target_pointer_width = "64")] const POINTER_WIDTH_BYTES: usize = 8;

    #[test]
    pub fn test_invalid_datatype() {
        unsafe {
            assert_err!(Datatype::from_id(H5I_INVALID_HID),
                        "Invalid datatype id");
            assert_err!(Datatype::from_id(h5lock!(H5Tcopy(*H5T_STD_REF_OBJ))),
                        "Unsupported datatype");
        }
    }

    #[test]
    pub fn test_eq() {
        assert_eq!(u32::to_datatype().unwrap(), u32::to_datatype().unwrap());
        assert_ne!(u32::to_datatype().unwrap(), u16::to_datatype().unwrap());
    }

    #[test]
    pub fn test_atomic_datatype() {
        fn test_integer<T: ToDatatype>(signed: bool, precision: usize, size: usize) {
            match <T as ToDatatype>::to_datatype().unwrap().class().unwrap() {
                DatatypeClass::Integer(dt) => {
                    assert_eq!(dt.is_be(), IS_BE);
                    assert_eq!(dt.is_le(), IS_LE);
                    assert_eq!(dt.offset(), 0);
                    assert_eq!(dt.precision(), precision);
                    assert_eq!(dt.is_signed(), signed);
                    assert_eq!(dt.size(), size);
                },
                _ => panic!("Integer datatype expected")
            }
        }

        fn test_float<T: ToDatatype>(precision: usize, size: usize) {
            match <T as ToDatatype>::to_datatype().unwrap().class().unwrap() {
                DatatypeClass::Float(dt) => {
                    assert_eq!(dt.is_be(), IS_BE);
                    assert_eq!(dt.is_le(), IS_LE);
                    assert_eq!(dt.offset(), 0);
                    assert_eq!(dt.precision(), precision);
                    assert_eq!(dt.size(), size);
                },
                _ => panic!("Float datatype expected")
            }
        }        test_integer::<bool>(false, 8, 1);

        test_integer::<i8>(true, 8, 1);
        test_integer::<i16>(true, 16, 2);
        test_integer::<i32>(true, 32, 4);
        test_integer::<i64>(true, 64, 8);

        test_integer::<u8>(false, 8, 1);
        test_integer::<u16>(false, 16, 2);
        test_integer::<u32>(false, 32, 4);
        test_integer::<u64>(false, 64, 8);

        test_float::<f32>(32, 4);
        test_float::<f64>(64, 8);

        test_integer::<isize>(true, POINTER_WIDTH_BYTES * 8, POINTER_WIDTH_BYTES);
        test_integer::<usize>(false, POINTER_WIDTH_BYTES * 8, POINTER_WIDTH_BYTES);
    }

    #[test]
    pub fn test_debug_display() {
        assert_eq!(format!("{}", u32::to_datatype().unwrap()),
            "<HDF5 datatype: 32-bit unsigned integer>");
        assert_eq!(format!("{:?}", u32::to_datatype().unwrap()),
            "<HDF5 datatype: 32-bit unsigned integer>");

        assert_eq!(format!("{}", i8::to_datatype().unwrap()),
            "<HDF5 datatype: 8-bit signed integer>");
        assert_eq!(format!("{}", i8::to_datatype().unwrap()),
            "<HDF5 datatype: 8-bit signed integer>");

        assert_eq!(format!("{}", f64::to_datatype().unwrap()),
            "<HDF5 datatype: 64-bit float>");
        assert_eq!(format!("{:?}", f64::to_datatype().unwrap()),
            "<HDF5 datatype: 64-bit float>");
    }
}
