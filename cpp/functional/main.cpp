#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <numeric>
//#include <execution>
//#include <experimental/numeric>
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

double average_score(const std::vector<int>& scores) {
    return std::accumulate(scores.cbegin(), scores.cend(), 0) / (double)scores.size();
}

//double average_score_reduce(const std::vector<int>& scores) {
//    return std::experimental::reduce(std::experimental::execution::par, scores.cbegin(), scores.cend(), 0) / (double)scores.size();
//    return std::reduce(scores.cbegin(), scores.cend(), 0) / (double)scores.size();
//}

double scores_product(const std::vector<int>& scores) {
    return std::accumulate(scores.cbegin(), scores.cend(), 1, std::multiplies<int>());
}

int main() {
    std::vector<std::string> files = {"file1.txt", "file2.txt"};
    auto res = count_lines_in_files(files);
    std::cout << res[0] << " " << res[1] << std::endl;

    auto f = [] <typename T> (T first, T second) { return first < second;};
    std::cout << "res " << f(10,11) << std::endl;

    std::vector<int> v {1,10,3,4,5,60,54};
    std::sort(v.begin(), v.end(), std::greater<>());
    std::cout << v[0];
    return 0;
}
