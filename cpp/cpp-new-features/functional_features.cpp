//
// Created by rostam on 14.10.19.
//

#include <vector>
#include <numeric>
//#include <execution>

double average_score(const std::vector<int>& scores)
{
    int sum = 0;

    for (int score : scores) {
        sum += score;
    }

    return sum / (double)scores.size();
}

double average_score_accumulate(const std::vector<int>& scores)
{
    return std::accumulate(
            scores.cbegin(), scores.cend(),
            0
    ) / (double)scores.size();
}
//
//double average_score_reduce(const std::vector<int>& scores)
//{
//    return std::reduce(
//            std::execution::par,
//            scores.cbegin(), scores.cend(),
//            0
//    ) / (double) scores.length();
//}

int main() {

}
