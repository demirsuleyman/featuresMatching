#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <vector>

using namespace cv;
using namespace std;

int main() {
    // Ana görüntüyü içe aktar
    Mat chocolates = imread("chocolates.jpg", IMREAD_GRAYSCALE);
    if(chocolates.empty()) {
        cout << "The main image could not be loaded" << endl;
        return -1;
    }
    imshow("Main Image", chocolates);

    // Aranacak görüntüyü içe aktar
    Mat chocolate = imread("nestle.jpg", IMREAD_GRAYSCALE);
    if(chocolate.empty()) {
        cout << "The search image could not be loaded." << endl;
        return -1;
    }
    imshow("Search Image", chocolate);

    // ORB Eşleştirme
    Ptr<ORB> orb = ORB::create();
    vector<KeyPoint> orb_kp1, orb_kp2;
    Mat orb_des1, orb_des2;

    // Anahtar nokta tespiti
    orb->detectAndCompute(chocolate, noArray(), orb_kp1, orb_des1);
    orb->detectAndCompute(chocolates, noArray(), orb_kp2, orb_des2);

    // BFMatcher ile eşleştirme
    BFMatcher bf(NORM_HAMMING);
    vector<DMatch> matches;
    bf.match(orb_des1, orb_des2, matches);

    // Mesafeye göre sıralama
    sort(matches.begin(), matches.end());

    // İlk 20 eşleşmeyi göster
    Mat img_match;
    drawMatches(chocolate, orb_kp1, chocolates, orb_kp2,
                vector<DMatch>(matches.begin(), matches.begin() + min(20, (int)matches.size())),
                img_match,
                Scalar::all(-1),
                Scalar::all(-1),
                std::vector<char>(),
                DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    imshow("ORB Matches", img_match);

    // SIFT Eşleştirme
    Ptr<SIFT> sift = SIFT::create();
    vector<KeyPoint> sift_kp1, sift_kp2;
    Mat sift_des1, sift_des2;

    // SIFT ile anahtar nokta tespiti
    sift->detectAndCompute(chocolate, noArray(), sift_kp1, sift_des1);
    sift->detectAndCompute(chocolates, noArray(), sift_kp2, sift_des2);

    // BFMatcher ile kNN eşleştirme
    BFMatcher bf_sift(NORM_L2);
    vector<vector<DMatch>> knn_matches;
    bf_sift.knnMatch(sift_des1, sift_des2, knn_matches, 2);

    // İyi eşleşmeleri seç
    vector<DMatch> good_matches;
    for(size_t i = 0; i < knn_matches.size(); i++) {
        if(knn_matches[i][0].distance < 0.75 * knn_matches[i][1].distance) {
            good_matches.push_back(knn_matches[i][0]);
        }
    }

    // SIFT eşleşmelerini göster
    Mat sift_matches;
    // Mask vektörünü doğru boyutta oluştur
    std::vector<char> mask(good_matches.size(), 1);

    drawMatches(chocolate, sift_kp1,
                chocolates, sift_kp2,
                good_matches,    // vector<DMatch> tipinde
                sift_matches,    // çıktı görüntüsü
                Scalar::all(-1), // eşleşme rengi
                Scalar::all(-1), // tek nokta rengi
                mask,
                DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS); // bayraklar

    imshow("SIFT Matches", sift_matches);

    // Herhangi bir tuşa basılmasını bekle
    waitKey(0);
    destroyAllWindows();

    return 0;
}