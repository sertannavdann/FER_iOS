#include "ProbabilityAdjustment.h"
#include "CoreMLBridge.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <deque>
#include <random>
#include <algorithm>

using namespace std;
using namespace cv;

const vector<string> classes = { "fear", "angry", "sad", "neutral", "surprise", "disgust", "happy" };
constexpr int64_t imageHeight = 128, imageWidth = 128, numClasses = 7;
constexpr int maxHistory = 60;
const string window_name = "Face Detection";

// Generate random colors for each class.
vector<Scalar> randomColors(size_t numColors) {
    vector<Scalar> colors;
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, 255);
    for (size_t i = 0; i < numColors; i++)
        colors.emplace_back(dis(gen), dis(gen), dis(gen));
    return colors;
}

// Apply exponential moving average smoothing.
vector<float> applyEma(const vector<float>& new_probs, vector<float>& ema_state, float alpha) {
    if (ema_state.empty()) {
        ema_state = new_probs;
        return ema_state;
    }
    for (size_t i = 0; i < new_probs.size(); ++i) {
        ema_state[i] = alpha * new_probs[i] + (1.0f - alpha) * ema_state[i];
    }
    return ema_state;
}

// Compute median probability per class over the history buffer.
vector<float> medianFromHistory(const deque<vector<float>>& history) {
    if (history.empty()) return {};
    size_t classesCount = history.front().size();
    vector<float> medians(classesCount, 0.0f);
    for (size_t c = 0; c < classesCount; ++c) {
        vector<float> col;
        col.reserve(history.size());
        for (const auto& h : history) {
            if (c < h.size()) col.push_back(h[c]);
        }
        if (!col.empty()) {
            nth_element(col.begin(), col.begin() + col.size() / 2, col.end());
            medians[c] = col[col.size() / 2];
        }
    }
    return medians;
}

// Run inference on the given face using the provided Core ML predictor.
vector<float> face2Int(Mat face, FERPredictor& predictor) {
    vector<float> probabilities;
    if (face.empty()) {
        cerr << "Input face is empty.\n";
        return probabilities;
    }
    if (face.rows != imageHeight || face.cols != imageWidth)
        resize(face, face, Size(imageWidth, imageHeight));
    
    // Ensure 3 channels
    if (face.channels() == 1) {
        cvtColor(face, face, COLOR_GRAY2BGR);
    }

    probabilities = predictor.predict(face);
    return probabilities;
}

// Draw detected face rectangles.
void drawDetectedFeatures(Mat& image, const vector<Rect>& features) {
    for (const auto& f : features)
        rectangle(image, f, Scalar(0, 255, 0), 2);
}

// Visualize probabilities as line graphs with history.
void visualizeProbabilities(const vector<float>& probabilities, const vector<string>& classes,
    Mat& graph, const deque<vector<float>>& history, int maxHistory, const vector<Scalar>& randomColorsVec) {
    int width = 1600, height = 400, margin = 5;
    int sectionWidth = width / classes.size();
    float scaleX = static_cast<float>(sectionWidth - 2 * margin) / maxHistory;
    float scaleY = static_cast<float>(height - 2 * margin);

    if (graph.empty())
        graph = Mat::zeros(height, width, CV_8UC3);
    graph.setTo(Scalar(255, 255, 255));

    for (int j = 0; j < classes.size(); j++) {
        int sectionStart = j * sectionWidth;
        line(graph, Point(sectionStart + margin, margin), Point(sectionStart + margin, height - margin),
            Scalar(0, 0, 0), 1, LINE_AA);
        line(graph, Point(sectionStart + margin, height - margin), Point(sectionStart + sectionWidth - margin, height - margin),
            Scalar(0, 0, 0), 1, LINE_AA);
        putText(graph, classes[j], Point(sectionStart + margin + 5, margin + 10),
            FONT_HERSHEY_PLAIN, 1, Scalar(0, 0, 0), 1, LINE_AA);
        for (size_t i = 0; i < history.size() - 1; i++) {
            Point pt1(static_cast<int>(sectionStart + margin + i * scaleX),
                static_cast<int>(height - margin - history[i][j] * scaleY));
            Point pt2(static_cast<int>(sectionStart + margin + (i + 1) * scaleX),
                static_cast<int>(height - margin - history[i + 1][j] * scaleY));
            line(graph, pt1, pt2, randomColorsVec[j], 1, LINE_AA);
        }
    }
    putText(graph, "DeltaTime", Point(width / 2 - 30, height - margin + 30),
        FONT_HERSHEY_PLAIN, 0.3, Scalar(0, 0, 0), 1, LINE_AA);
    putText(graph, "Probability", Point(width - 60, margin - 10),
        FONT_HERSHEY_PLAIN, 0.3, Scalar(0, 0, 0), 1, LINE_AA);
    imshow("Probabilities", graph);
}

// Capture video, perform face detection, inference, and visualization.
void captureVideoAndProcess(const string& cascadePath, const string& modelPath) {
    CascadeClassifier classifier;
    if (!classifier.load(cascadePath)) {
        cerr << "Error loading cascade from: " << cascadePath << "\n";
        return;
    }
    VideoCapture capture(0);
    if (!capture.isOpened()) {
        cerr << "Cannot open video capture device\n";
        return;
    }

    // Create Core ML Predictor.
    FERPredictor predictor(modelPath);

    Mat image, gray;
    vector<Rect> features;
    deque<vector<float>> history;
    vector<float> ema_state;
    const float ema_alpha = 0.1f;
    Mat graph;
    vector<Scalar> randomColorsVec = randomColors(classes.size());
    namedWindow("Probabilities", WINDOW_NORMAL);

    while (capture.read(image) && !image.empty()) {
        cvtColor(image, gray, COLOR_BGR2GRAY);
        equalizeHist(gray, gray);
        classifier.detectMultiScale(gray, features, 1.1, 5, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
        drawDetectedFeatures(image, features);
        if (!features.empty()) {
            Mat face = gray(features[0]);
            vector<float> probabilities = face2Int(face, predictor);
            if (probabilities.empty()) {
                imshow(window_name, image);
                continue;
            }

            // Adjust probabilities (e.g., boost neutral) then smooth.
            EmotionProbabilityAdjuster adjuster(2.0f);
            vector<float> adjustedProbabilities = adjuster.adjust(probabilities);

            vector<float> ema_probs = applyEma(adjustedProbabilities, ema_state, ema_alpha);
            history.push_back(ema_probs);
            if (history.size() > maxHistory) history.pop_front();

            vector<float> median_probs = medianFromHistory(history);
            const vector<float>& display_probs = median_probs.empty() ? ema_probs : median_probs;

            visualizeProbabilities(display_probs, classes, graph, history, maxHistory, randomColorsVec);
        }
        imshow(window_name, image);
        char key = (char)waitKey(10);
        if (key == 'q' || key == 'Q')
            break;
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <cascade.xml> <model.mlpackage>\n";
        return EXIT_FAILURE;
    }
    captureVideoAndProcess(argv[1], argv[2]);
    return EXIT_SUCCESS;
}
