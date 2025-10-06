use uuid::Uuid;

fn main() {
    // Test which UUID methods are available
    let id = Uuid::new_v4();
    println!("UUID: {}", id);
}