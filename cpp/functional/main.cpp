#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
//#include <ranges>

int count_lines(const std::string& filename) {
    std::ifstream in(filename);
    return std::count(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>(), '\n');
}

std::vector<int> count_lines_in_files(const std::vector<std::string>& files) {
    std::vector<int> results(files.size());
    std::transform(files.cbegin(), files.cend(), results.begin(), count_lines);
    return results;
}

int main() {
    std::vector<std::string> files = {"file1.txt", "file2.txt"};
    auto res = count_lines_in_files(files);
    std::cout << res[0] << " " << res[1] << std::endl;
    return 0;
}
