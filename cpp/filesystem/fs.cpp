#include <iostream>
#include <filesystem>

int main() {
  std::cout << std::filesystem::current_path() << std::endl;
  std::cout << std::filesystem::temp_directory_path() << std::endl;
  std::cout << std::filesystem::path("/home/rostam").filename() << std::endl;
  std::cout << __FILE__ << std::endl;
  std::cout << std::filesystem::path(__FILE__).remove_filename() << std::endl;

  std::filesystem::create_directories("sandbox/a/b");
  return 0;
}
 