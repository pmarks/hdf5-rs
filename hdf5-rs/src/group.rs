use ffi::h5i::{H5I_GROUP, hid_t};

use error::Result;
use object::{Object, ObjectType, AllowTypes};
use container::ContainerType;
use location::LocationType;

use std::fmt;

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
}

/// Represents the HDF5 group object.
pub type Group = Object<GroupID>;

impl LocationType for GroupID {}
impl ContainerType for GroupID {}

impl fmt::Debug for Group {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for Group {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if !self.is_valid() {
            return "<HDF5 group: invalid id>".fmt(f);
        }
        let members = match self.len() {
            0 => "empty".to_owned(),
            1 => "1 member".to_owned(),
            x => format!("{} members", x),
        };
        format!("<HDF5 group: \"{}\" ({})>", self.name(), members).fmt(f)
    }
}

#[cfg(test)]
pub mod tests {
    use test::with_tmp_file;

    #[test]
    pub fn test_debug_display() {
        with_tmp_file(|file| {
            file.create_group("a/b/c").unwrap();
            file.create_group("/a/d").unwrap();
            let a = file.group("a").unwrap();
            let ab = file.group("/a/b").unwrap();
            let abc = file.group("./a/b/c/").unwrap();
            assert_eq!(format!("{}", a), "<HDF5 group: \"/a\" (2 members)>");
            assert_eq!(format!("{:?}", a), "<HDF5 group: \"/a\" (2 members)>");
            assert_eq!(format!("{}", ab), "<HDF5 group: \"/a/b\" (1 member)>");
            assert_eq!(format!("{:?}", ab), "<HDF5 group: \"/a/b\" (1 member)>");
            assert_eq!(format!("{}", abc), "<HDF5 group: \"/a/b/c\" (empty)>");
            assert_eq!(format!("{:?}", abc), "<HDF5 group: \"/a/b/c\" (empty)>");
            file.close();
            assert_eq!(format!("{}", a), "<HDF5 group: invalid id>");
            assert_eq!(format!("{:?}", a), "<HDF5 group: invalid id>");
        })
    }
}
