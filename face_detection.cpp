#include "face_detection.h"

int main(int argc, char* argv[]) {
    if (argc != 2) {
        cerr << "usage: " << argv[0] << " classifier.xml\n";
        exit(EXIT_FAILURE);
    }

    CascadeClassifier classifier(argv[1]);
    captureVideoAndProcess(classifier);

    return EXIT_SUCCESS;
}

void captureVideoAndProcess(CascadeClassifier& classifier) {
    const char* const window_name{ "Facial Recognition Window" };
    VideoCapture capture(0);
    if (not capture.isOpened()) {
        cerr << "cannot open video capture device\n";
        exit(EXIT_FAILURE);
    }

    Mat image;
    Mat grayscale_image;
    vector<Rect> features;
    Mat face;

    deque<vector<float>> history;
    Mat graph;
    vector<Scalar> randomColorsVec = randomColors(numClasses);

    namedWindow("Probabilities", WINDOW_NORMAL);

    while (capture.read(image) and (not image.empty())) {
        cvtColor(image, grayscale_image, COLOR_BGR2GRAY);
        equalizeHist(grayscale_image, grayscale_image);

        classifier.detectMultiScale(grayscale_image, features, 1.1, 5, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

        drawDetectedFeatures(image, features);

        if (!features.empty()) {
            // Extract the detected face from the grayscale image
            face = grayscale_image(features[0]);

            vector<float> probabilities = face2Int(face);
            // Generate random colors for each class

            // Call the visualizeProbabilities function with the randomColorsVec
            visualizeProbabilities(probabilities, classes, graph, history, maxHistory, randomColorsVec);
        }

        imshow(window_name, image);

        switch (waitKey(10)) {
        case 'q':
            exit(EXIT_SUCCESS);
        case 'Q':
            exit(EXIT_SUCCESS);
        default:
            break;
        }
    }
}

void drawDetectedFeatures(Mat& image, const vector<Rect>& features) {
    for (auto&& feature : features) {
        rectangle(image, feature, Scalar(0, 255, 0), 2);
    }
}

void visualizeProbabilities(const vector<float>& probabilities, const vector<string>& classes, Mat& graph, deque<vector<float>>& history, int maxHistory, const vector<Scalar>& randomColorsVec)
{
    int width = 1200;
    int height = 400;
    int numClasses = static_cast<int>(classes.size());
    int margin = 5;
    int sectionWidth = width / numClasses;
    float scaleX = static_cast<float>(sectionWidth - 2 * margin) / maxHistory;
    float scaleY = static_cast<float>(height - 2 * margin);

    if (graph.empty()) {
        graph = Mat::zeros(height, width, CV_8UC3);
    }

    history.push_back(probabilities);

    if (history.size() > maxHistory) {
        history.pop_front();
    }

    // Clear graph
    graph.setTo(Scalar(255, 255, 255));

    for (int j = 0; j < numClasses; ++j) {
        int sectionStart = j * sectionWidth;

        // Draw axis
        line(graph, Point(sectionStart + margin, margin), Point(sectionStart + margin, height - margin), Scalar(0, 0, 0), 1, LINE_AA);
        line(graph, Point(sectionStart + margin, height - margin), Point(sectionStart + sectionWidth - margin, height - margin), Scalar(0, 0, 0), 1, LINE_AA);

        // Draw class name label
        putText(graph, classes[j], Point(sectionStart + margin+5, margin+10), FONT_HERSHEY_PLAIN, 1, Scalar(0, 0, 0), 1, LINE_AA);

        // Draw lines
        for (size_t i = 0; i < history.size() - 1; ++i) {
            Point pt1(static_cast<int>(sectionStart + margin + i * scaleX), static_cast<int>(height - margin - history[i][j] * scaleY));
            Point pt2(static_cast<int>(sectionStart + margin + (i + 1) * scaleX), static_cast<int>(height - margin - history[i + 1][j] * scaleY));
            
            // Use the random color for each line
            Scalar color = randomColorsVec[j];

            line(graph, pt1, pt2, color, 1, LINE_AA);
        }
    }

    // Draw "DeltaTime" label
    putText(graph, "DeltaTime", Point(width / 2 - 30, height - margin + 30), FONT_HERSHEY_PLAIN, 0.3, Scalar(0, 0, 0), 1, LINE_AA);

    // Draw "Probability" label on the top right corner
    putText(graph, "Probability", Point(width - 60, margin - 10), FONT_HERSHEY_PLAIN, 0.3, Scalar(0, 0, 0), 1, LINE_AA);

    imshow("Probabilities", graph);
}


vector<float> face2Int(Mat face) {
    vector<float> probabilities;

    // Check if the input face is empty
    if (face.empty()) {
        cerr << "Input face is empty. Skipping processing." << endl;
        return probabilities;
    }

    if (face.rows != 128 || face.cols != 128) {
        resize(face, face, Size(128, 128));
    }

    // Convert the grayscale face image back to a three-channel image
    cvtColor(face, face, COLOR_GRAY2BGR);

    Mat face_input;
    face.convertTo(face_input, CV_32F);
    face_input = face_input.reshape(1, { 1, 3, 128, 128 });
    face_input /= 255.0;


    Ort::Env env{ ORT_LOGGING_LEVEL_WARNING, "FER_Model" };
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    Ort::Session session(env, model_path, session_options);

    Ort::RunOptions runOptions;

    Ort::MemoryInfo allocator_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        allocator_info, face_input.ptr<float>(), face_input.total(),
        session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape().data(),
        session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape().size()
    );

    vector<const char*> input_names = { input_name.c_str() };
    vector<const char*> output_names = { output_name.c_str() };

    // define shape
    const vector<int64_t> inputShape = { 1, 3, 128, 128 };
    const vector<int64_t> outputShape = { 1, 7 };

    // define array
    auto input = make_shared<vector<float>>(numInputElements);
    auto results = make_shared<vector<float>>(numClasses);

    // define Tensor
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    auto inputTensor = Ort::Value::CreateTensor<float>(memory_info, input->data(), input->size(), inputShape.data(), inputShape.size());
    auto outputTensor = Ort::Value::CreateTensor<float>(memory_info, results->data(), results->size(), outputShape.data(), outputShape.size());

    vector<float> imageVec = loadImage(face, 128, 128);
    // copy image data to input array
    copy(imageVec.begin(), imageVec.end(), input->begin());

    // define names
    Ort::AllocatorWithDefaultOptions ort_alloc;
    Ort::AllocatedStringPtr inputName = session.GetInputNameAllocated(0, ort_alloc);
    Ort::AllocatedStringPtr outputName = session.GetOutputNameAllocated(0, ort_alloc);

    const array<const char*, 1> inputNames = { inputName.get() };
    const array<const char*, 1> outputNames = { outputName.get() };

    inputName.release();
    outputName.release();

    // run inference
    try {
        session.Run(runOptions, inputNames.data(), &inputTensor, 1, outputNames.data(), &outputTensor, 1);
    }
    catch (Ort::Exception& e) {
        cout << e.what() << endl;
        return probabilities;
    }
    // sort results
    vector<pair<size_t, float>> indexValuePairs;
    for (size_t i = 0; i < results->size(); ++i) {
        indexValuePairs.emplace_back(i, (*results)[i]);
    }
    sort(indexValuePairs.begin(), indexValuePairs.end(), [](const auto& lhs, const auto& rhs) { return lhs.second > rhs.second; });

    // Add all probabilities to the return vector
    for (const auto& result : *results) {
        probabilities.push_back(result);
    }

    return probabilities;
}

vector<float> loadImage(Mat image, int sizeX = 128, int sizeY = 128)
{
    if (image.empty()) {
        cout << "No image found.";
    }

    // convert from BGR to RGB
    cvtColor(image, image, COLOR_BGR2RGB);

    // resize
    resize(image, image, Size(sizeX, sizeY));

    // reshape to 1D
    image = image.reshape(1, 1);

    // uint_8, [0, 255] -> float, [0, 1]
    vector<float> vec;
    image.convertTo(vec, CV_32FC1, 1. / 255);

    // Transpose (Height, Width, Channel)(224,224,3) to (Chanel, Height, Width)(3,224,224)
    vector<float> output;
    for (size_t ch = 0; ch < 3; ++ch) {
        for (size_t i = ch; i < vec.size(); i += 3) {
            output.emplace_back(vec[i]);
        }
    }
    return output;
}

// Generate random colors for each class
vector<Scalar> randomColors(int numColors) {
    vector<Scalar> colors;
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, 255);
    for (int i = 0; i < numColors; i++) {
        Scalar color(dis(gen), dis(gen), dis(gen));
        colors.push_back(color);
    }
    return colors;
}