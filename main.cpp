#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <vector>

using namespace cv;
using namespace std;

int main() {
    // Load the main image (scene) in grayscale
    Mat chocolates = imread("chocolates.jpg", IMREAD_GRAYSCALE);
    if(chocolates.empty()) {
        cout << "The main image could not be loaded" << endl;
        return -1;
    }
    // Display the main image
    imshow("Main Image", chocolates);

    // Load the image to search for (object) in grayscale
    Mat chocolate = imread("nestle.jpg", IMREAD_GRAYSCALE);
    if(chocolate.empty()) {
        cout << "The search image could not be loaded." << endl;
        return -1;
    }
    // Display the search image
    imshow("Search Image", chocolate);

    // ORB (Oriented FAST and Rotated BRIEF) keypoint detection and matching
    // Create ORB detector
    Ptr<ORB> orb = ORB::create();
    vector<KeyPoint> orb_kp1, orb_kp2; // Keypoints vectors for the two images
    Mat orb_des1, orb_des2; // Descriptors matrices for the two images

    // Detect keypoints and compute descriptors for the search image
    orb->detectAndCompute(chocolate, noArray(), orb_kp1, orb_des1);
    // Detect keypoints and compute descriptors for the main image
    orb->detectAndCompute(chocolates, noArray(), orb_kp2, orb_des2);

    // Brute-force matcher using Hamming distance (suitable for binary descriptors like ORB)
    BFMatcher bf(NORM_HAMMING);
    vector<DMatch> matches; // Store matches between the two images
    bf.match(orb_des1, orb_des2, matches); // Match descriptors between the two images

    // Sort matches based on distance (lower distance means better match)
    sort(matches.begin(), matches.end());

    // Draw the top 20 matches based on the sorted distances
    Mat img_match;
    drawMatches(chocolate, orb_kp1, chocolates, orb_kp2,
                vector<DMatch>(matches.begin(), matches.begin() + min(20, (int)matches.size())),
                img_match,        // Output image to store the matches
                Scalar::all(-1),  // Match line color (random by default)
                Scalar::all(-1),  // Single keypoint color (random by default)
                vector<char>(),   // No mask, draw all matches
                DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS); // Do not draw unmatched keypoints
    imshow("ORB Matches", img_match); // Display the ORB matches

    // SIFT (Scale-Invariant Feature Transform) keypoint detection and matching
    // Create SIFT detector
    Ptr<SIFT> sift = SIFT::create();
    vector<KeyPoint> sift_kp1, sift_kp2; // Keypoints vectors for SIFT
    Mat sift_des1, sift_des2; // Descriptors matrices for SIFT

    // Detect keypoints and compute descriptors for the search image using SIFT
    sift->detectAndCompute(chocolate, noArray(), sift_kp1, sift_des1);
    // Detect keypoints and compute descriptors for the main image using SIFT
    sift->detectAndCompute(chocolates, noArray(), sift_kp2, sift_des2);

    // Brute-force matcher using L2 norm (suitable for non-binary descriptors like SIFT)
    BFMatcher bf_sift(NORM_L2);
    vector<vector<DMatch>> knn_matches; // Store the top 2 matches for each keypoint (k-NN)
    bf_sift.knnMatch(sift_des1, sift_des2, knn_matches, 2); // kNN matching with k=2

    // Select good matches using the ratio test (Lowe's ratio test)
    vector<DMatch> good_matches;
    for(size_t i = 0; i < knn_matches.size(); i++) {
        // Check if the first match is significantly better than the second match
        if(knn_matches[i][0].distance < 0.75 * knn_matches[i][1].distance) {
            good_matches.push_back(knn_matches[i][0]); // Keep the good match
        }
    }

    // Draw good SIFT matches
    Mat sift_matches;
    drawMatches(chocolate, sift_kp1,
                chocolates, sift_kp2,
                good_matches,      // Vector of good matches to display
                sift_matches,      // Output image to store the matches
                Scalar::all(-1),   // Match line color (random by default)
                Scalar::all(-1),   // Single keypoint color (random by default)
                vector<char>(),    // No mask, draw all good matches
                DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS); // Do not draw unmatched keypoints
    imshow("SIFT Matches", sift_matches); // Display the SIFT matches

    // Wait indefinitely until a key is pressed
    waitKey(0);
    // Destroy all windows after key press
    destroyAllWindows();

    return 0;
}
