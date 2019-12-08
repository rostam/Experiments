#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>

int count_lines(const std::string& filename)
{
    std::ifstream in(filename);
    return std::count(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>(in), '\n');
}

std::vector<int> count_lines_in_files(const std::vector<std::string> files)
{
    return files | transform(count_lines)
}

int main() {
    auto res = count_lines_in_files(std::vector<int>({'file.txt','file2.txt'}));
    std::cout << res[0] << " " << res[1] << std::endl;
    return 0;
}
