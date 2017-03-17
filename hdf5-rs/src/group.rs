use ffi::h5i::{H5I_GROUP, hid_t};

use error::Result;
use object::{Object, ObjectType, AllowTypes};
use container::ContainerType;
use location::LocationType;

pub struct GroupID;

impl ObjectType for GroupID {
    fn allow_types() -> AllowTypes {
        AllowTypes::Just(H5I_GROUP)
    }

    fn from_id(_: hid_t) -> Result<GroupID> {
        Ok(GroupID)
    }

    fn type_name() -> &'static str {
        "group"
    }

    fn describe(obj: &Group) -> String {
        let members = match obj.len() {
            0 => "empty".to_owned(),
            1 => "1 member".to_owned(),
            x => format!("{} members", x),
        };
        // FIXME: anonymous groups -> <anonymous>
        format!("\"{}\" ({})", obj.name(), members)
    }
}

/// Represents the HDF5 group object.
pub type Group = Object<GroupID>;

impl LocationType for GroupID {}
impl ContainerType for GroupID {}

#[cfg(test)]
pub mod tests {
    use test::with_tmp_file;

    #[test]
    pub fn test_debug() {
        with_tmp_file(|file| {
            file.create_group("a/b/c").unwrap();
            file.create_group("/a/d").unwrap();
            let a = file.group("a").unwrap();
            let ab = file.group("/a/b").unwrap();
            let abc = file.group("./a/b/c/").unwrap();
            assert_eq!(format!("{:?}", a), "<HDF5 group: \"/a\" (2 members)>");
            assert_eq!(format!("{:?}", ab), "<HDF5 group: \"/a/b\" (1 member)>");
            assert_eq!(format!("{:?}", abc), "<HDF5 group: \"/a/b/c\" (empty)>");
            file.close();
            assert_eq!(format!("{:?}", a), "<HDF5 group: invalid id>");
        })
    }
}
