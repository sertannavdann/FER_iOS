//
//  ProbabilityAdjustment.h
//  FacialExpressionDetection
//
//  Created by Sertan Avdan on 2025-03-26.
//

#include <vector>
#include <numeric>

class EmotionProbabilityAdjuster {
private:
    float boostFactor;  // Factor to boost the neutral probability.
    int neutralIndex;   // Index of the neutral class (default 3).

public:
    // Constructor with default boostFactor = 1.5 and neutralIndex = 3.
    EmotionProbabilityAdjuster(float boostFactor = 1.5f, int neutralIndex = 3)
        : boostFactor(boostFactor), neutralIndex(neutralIndex) {}

    // Adjust the probabilities by boosting the neutral class and re-normalizing.
    std::vector<float> adjust(const std::vector<float>& probabilities) {
        std::vector<float> adjusted = probabilities;
        if (adjusted.size() > static_cast<size_t>(neutralIndex)) {
            adjusted[neutralIndex] *= boostFactor;
        }
        // Re-normalize the probabilities so they sum to 1.
        float sum = std::accumulate(adjusted.begin(), adjusted.end(), 0.0f);
        if (sum > 0) {
            for (auto &p : adjusted)
                p /= sum;
        }
        return adjusted;
    }
};
