#include "Params.h"


/** Default parameters */
tld::Params::Params()
{
    // Random generator seed
    RNG_SEED = 42;

    // Median Flow tracker parameters
    LEN_POINTS = 10;
    TOTAL_NUM_POINTS = LEN_POINTS * LEN_POINTS; 
    RAND_POINTS = true;
    LK_WIN_SIZE = cv::Size(15, 15);
    MAX_PYR_LEVEL = 3;
    NCC_PATCH_SIZE = cv::Size(10, 10);
    FB_THRESHOLD = 10.0f;
    MAX_MEDIAN_DISPLACEMENT = 10.0f;
    TERM_CRITERIA = cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 10, 0.03);

    // Cascade classifier parameters
    VARIANCE_FRACTION = 0.7f;
    OVERLAP_THRESHOLD = 0.5f;
    NUM_FERNS = 5;
    NUM_BINARY_FEATURES = 4;
    SCALE_STEP = 0.2f;
    MIN_SCALE = 0.2f;
    MAX_SCALE = 2.2f;
    WIDTH_FRACTION = MIN_SCALE / 2.0f;
    HEIGHT_FRACTION = MIN_SCALE / 2.0f;
    MIN_AREA = 25.0f;

    // TLD parameters
    THETA_MINUS = 0.65f;
    THETA_PLUS = 0.7f;

    // Object model parameters
    RAND_REPLACEMENT = false;
    TEMPLATE_SIZE = cv::Size(15, 15);
    INIT_OBJ_MODEL_SIZE = 20;
    MAX_OBJ_MODEL_SIZE = 40;
}


/**
 * Read parameters from a yaml file
 */
void tld::Params::read(const std::string& filename)
{
    cv::FileStorage fs(filename, cv::FileStorage::READ);

    if (!fs["RNG_SEED"].empty())
        RNG_SEED = fs["RNG_SEED"];

    // Median Flow tracker parameters
    if (!fs["LEN_POINTS"].empty())
        LEN_POINTS = fs["LEN_POINTS"];
    if (!fs["TOTAL_NUM_POINTS"].empty())
        TOTAL_NUM_POINTS = fs["TOTAL_NUM_POINTS"];
    if (!fs["RAND_POINTS"].empty())
        RAND_POINTS = (static_cast<int>(fs["RAND_POINTS"]) != 0);
    if (!fs["LK_WIN_SIZE"].empty())
        LK_WIN_SIZE = cv::Size(fs["LK_WIN_SIZE"][0], fs["LK_WIN_SIZE"][1]);
    if (!fs["MAX_PYR_LEVEL"].empty())
        MAX_PYR_LEVEL = fs["MAX_PYR_LEVEL"];
    if (!fs["NCC_PATCH_SIZE"].empty())
        NCC_PATCH_SIZE = cv::Size(fs["NCC_PATCH_SIZE"][0], fs["NCC_PATCH_SIZE"][1]);
    if (!fs["FB_THRESHOLD"].empty())
        FB_THRESHOLD = static_cast<float>(fs["FB_THRESHOLD"]);
    if (!fs["MAX_MEDIAN_DISPLACEMENT"].empty())
        MAX_MEDIAN_DISPLACEMENT = static_cast<float>(fs["MAX_MEDIAN_DISPLACEMENT"]);
    if (!fs["TERM_CRITERIA_COUNT"].empty())
        TERM_CRITERIA.maxCount = fs["TERM_CRITERIA_COUNT"];
    if (!fs["TERM_CRITERIA_EPS"].empty())
        TERM_CRITERIA.epsilon = static_cast<float>(fs["TERM_CRITERIA_EPS"]);
    
    // Cascade classifier parameters
    if (!fs["VARIANCE_FRACTION"].empty())
        VARIANCE_FRACTION = static_cast<float>(fs["VARIANCE_FRACTION"]);
    if (!fs["OVERLAP_THRESHOLD"].empty())
        OVERLAP_THRESHOLD = static_cast<float>(fs["OVERLAP_THRESHOLD"]);
    if (!fs["NUM_FERNS"].empty())
        NUM_FERNS = fs["NUM_FERNS"];
    if (!fs["NUM_BINARY_FEATURES"].empty())
        NUM_BINARY_FEATURES = fs["NUM_BINARY_FEATURES"];
    if (!fs["SCALE_STEP"].empty())
        SCALE_STEP = static_cast<float>(fs["SCALE_STEP"]);
    if (!fs["MIN_SCALE"].empty())
        MIN_SCALE = static_cast<float>(fs["MIN_SCALE"]);
    if (!fs["MAX_SCALE"].empty())
        MAX_SCALE = static_cast<float>(fs["MAX_SCALE"]);
    if (!fs["WIDTH_FRACTION"].empty())
        WIDTH_FRACTION = static_cast<float>(fs["WIDTH_FRACTION"]);
    if (!fs["HEIGHT_FRACTION"].empty())
        HEIGHT_FRACTION = static_cast<float>(fs["HEIGHT_FRACTION"]); 
    if (!fs["MIN_AREA"].empty())
        MIN_AREA = static_cast<float>(fs["MIN_AREA"]); 
    if (!fs["THETA_PLUS"].empty())
        THETA_PLUS = static_cast<float>(fs["THETA_PLUS"]);
    if (!fs["THETA_MINUS"].empty())
        THETA_MINUS = static_cast<float>(fs["THETA_MINUS"]);

    // Object model parameters
    if (!fs["RAND_REPLACEMENT"].empty())
        RAND_REPLACEMENT = (static_cast<int>(fs["RAND_REPLACEMENT"]) != 0);
    if (!fs["TEMPLATE_SIZE"].empty())
        TEMPLATE_SIZE = cv::Size(fs["TEMPLATE_SIZE"][0], fs["TEMPLATE_SIZE"][1]);
    if (!fs["INIT_OBJ_MODEL_SIZE"].empty())
        INIT_OBJ_MODEL_SIZE = static_cast<size_t>(static_cast<int>(fs["INIT_OBJ_MODEL_SIZE"]));
    if (!fs["MAX_OBJ_MODEL_SIZE"].empty())
        MAX_OBJ_MODEL_SIZE = static_cast<size_t>(static_cast<int>(fs["MAX_OBJ_MODEL_SIZE"]));
}


/**
 * Write parameters to a yaml file
 */
void tld::Params::write(const std::string& filename) const
{
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);

    fs << "RNG_SEED" << RNG_SEED;

    // Median Flow tracker parameters
    fs << "LEN_POINTS" << LEN_POINTS;
    fs << "TOTAL_NUM_POINTS" << TOTAL_NUM_POINTS; 
    fs << "RAND_POINTS" << RAND_POINTS;
    fs << "LK_WIN_SIZE" << LK_WIN_SIZE;
    fs << "MAX_PYR_LEVEL" << MAX_PYR_LEVEL;
    fs << "NCC_PATCH_SIZE" << NCC_PATCH_SIZE;
    fs << "FB_THRESHOLD" << FB_THRESHOLD;
    fs << "MAX_MEDIAN_DISPLACEMENT" << MAX_MEDIAN_DISPLACEMENT;
    fs << "TERM_CRITERIA_COUNT" << TERM_CRITERIA.maxCount;
    fs << "TERM_CRITERIA_EPS" << TERM_CRITERIA.epsilon;

    // Cascade classifier parameters
    fs << "VARIANCE_FRACTION" << VARIANCE_FRACTION;
    fs << "OVERLAP_THRESHOLD" << OVERLAP_THRESHOLD;
    fs << "NUM_FERNS" << NUM_FERNS;
    fs << "NUM_BINARY_FEATURES" << NUM_BINARY_FEATURES;
    fs << "SCALE_STEP" << SCALE_STEP;
    fs << "MIN_SCALE" << MIN_SCALE;
    fs << "MAX_SCALE" << MAX_SCALE;
    fs << "WIDTH_FRACTION" << WIDTH_FRACTION;
    fs << "HEIGHT_FRACTION" << HEIGHT_FRACTION;
    fs << "MIN_AREA" << MIN_AREA;
    fs << "THETA_PLUS" << THETA_PLUS;
    fs << "THETA_MINUS" << THETA_MINUS;

    // Object model parameters
    fs << "RAND_REPLACEMENT" << RAND_REPLACEMENT;
    fs << "TEMPLATE_SIZE" << TEMPLATE_SIZE;
    fs << "INIT_OBJ_MODEL_SIZE" << static_cast<int>(INIT_OBJ_MODEL_SIZE);
    fs << "MAX_OBJ_MODEL_SIZE" << static_cast<int>(MAX_OBJ_MODEL_SIZE);
    
}


/** Print all the parameters */
void tld::Params::printParams() const
{
    std::cout << "--------------------------------" << std::endl
              << "RNG_SEED: " << RNG_SEED << std::endl;

    std::cout << "--------------------------------" << std::endl
              << "Median Flow tracker parameters: " << std::endl
              << " LEN_POINTS: " << LEN_POINTS << std::endl
              << " TOTAL_NUM_POINTS: " << TOTAL_NUM_POINTS << std::endl
              << " RAND_POINTS: " << RAND_POINTS << std::endl
              << " LK_WIN_SIZE: " << LK_WIN_SIZE << std::endl
              << " MAX_PYR_LEVEL: " << MAX_PYR_LEVEL << std::endl
              << " NCC_PATCH_SIZE: " << NCC_PATCH_SIZE << std::endl
              << " FB_THRESHOLD: " << FB_THRESHOLD << std::endl
              << " MAX_MEDIAN_DISPLACEMENT: " << MAX_MEDIAN_DISPLACEMENT << std::endl
              << " TERM_CRITERIA_COUNT: " << TERM_CRITERIA.maxCount << std::endl
              << " TERM_CRITERIA_EPS: " << TERM_CRITERIA.epsilon << std::endl;

    std::cout << "--------------------------------" << std::endl
              << "Cascade classifier parameters: " << std::endl
              << " VARIANCE_FRACTION: " << VARIANCE_FRACTION << std::endl
              << " OVERLAP_THRESHOLD: " << OVERLAP_THRESHOLD << std::endl
              << " NUM_FERNS: " << NUM_FERNS << std::endl
              << " NUM_BINARY_FEATURES: " << NUM_BINARY_FEATURES << std::endl
              << " SCALE_STEP: " << SCALE_STEP << std::endl
              << " MIN_SCALE: " << MIN_SCALE << std::endl
              << " MAX_SCALE: " << MAX_SCALE << std::endl
              << " WIDTH_FRACTION: " << WIDTH_FRACTION << std::endl
              << " HEIGHT_FRACTION: " << HEIGHT_FRACTION << std::endl
              << " MIN_AREA: " << MIN_AREA << std::endl
              << " THETA_PLUS: " << THETA_PLUS << std::endl
              << " THETA_MINUS: " << THETA_MINUS << std::endl;

    std::cout << "--------------------------------" << std::endl
              << "Object model parameters: " << std::endl
              << " RAND_REPLACEMENT: " << RAND_REPLACEMENT << std::endl
              << " TEMPLATE_SIZE: " << TEMPLATE_SIZE << std::endl
              << " INIT_OBJ_MODEL_SIZE: " << INIT_OBJ_MODEL_SIZE << std::endl
              << " MAX_OBJ_MODEL_SIZE: " << MAX_OBJ_MODEL_SIZE << std::endl
              << "--------------------------------" << std::endl
              << std::endl;
}
