#include <pcl/io/io.h>
#include <pcl/io/ply_io.h> 
#include <pcl/point_types.h>
#include <pcl/octree/octree.h>
#include <pcl/filters/crop_box.h>
#include <iostream>
#include <sstream>
#include <pcl/point_cloud.h>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <pcl/PolygonMesh.h>
#include <algorithm>
#include <unordered_map>
#include <vector>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <ctime>
#include <nanoflann.hpp>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/random_sample.h>
#include <fstream>
#include <string>

using namespace std;

double f(double a) {

    // return (1 / (1 * (0.04 + a)));             
    // return (0.5 * a * log(a));        
    return (exp(-0.5 * a));
}

template <typename T>
struct RecPointCloud
{
    struct Point {
        T x, y, z;
    };

    std::vector<Point> points;  

    RecPointCloud() {
        
    }

    // Must return the number of data points
    inline size_t kdtree_get_point_count() const { return points.size(); }

    inline T kdtree_get_pt(const size_t idx, const size_t dim) const
    {
        if (dim == 0) return points[idx].x;
        else if (dim == 1) return points[idx].y;
        else return points[idx].z;
    }

    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /* bb */) const { return false; }

};

typedef pcl::PointXYZRGB PointType;
typedef std::vector< pcl::PointXYZRGB, Eigen::aligned_allocator<pcl::PointXYZRGB> > AlignedPointTVector;   

bool belongsToVector(PointType point, pcl::PointCloud <pcl::PointXYZRGB> vector) {
    for (int i = 0; i < vector.points.size(); i++) {
        if (vector.points[i].x == point.x && vector.points[i].y == point.y && vector.points[i].z == point.z) {
            return true;
        }
        return false;
    }
}

// ************************************  RBFParam *****************************************

struct rbfParam {
    std::vector<double> rbfA;
    std::vector<double> rbfB;
    std::vector<double> rbfC;
    std::vector<double> rbfD;
    std::vector<double> rbfOmega;

    rbfParam(
        std::vector<double> rbfa,
        std::vector<double> rbfb,
        std::vector<double> rbfc,
        std::vector<double> rbfd,
        std::vector<double> rbfomega
    ) {
        this->rbfA = rbfa;
        this->rbfB = rbfb;
        this->rbfC = rbfc;
        this->rbfD = rbfd;
        this->rbfOmega = rbfomega;
    };
};


using namespace nanoflann;
typedef KDTreeSingleIndexAdaptor<
    L2_Simple_Adaptor<float, RecPointCloud<float>>,
    RecPointCloud<float>,
    3 /* dimensions */
> KDTree;
void fitSurfaceAndCalculateCurvature(const pcl::PointCloud<PointType>::Ptr& cloud, std::vector<uint32_t> indices, double& meanCurvature) {
    int num_points = indices.size();
    Eigen::MatrixXd A(num_points, 6);
    Eigen::VectorXd B(num_points);

    for (int i = 0; i < num_points; ++i) {
        const auto& p = cloud->points[indices[i]];
        double x = p.x;
        double y = p.y;
        double z = p.z;
        A(i, 0) = x * x;
        A(i, 1) = y * y;
        A(i, 2) = x * y;
        A(i, 3) = x;
        A(i, 4) = y;
        A(i, 5) = 1;
        B(i) = z;
    }

    Eigen::VectorXd coeffs = A.colPivHouseholderQr().solve(B);

    double a = coeffs(0);
    double b = coeffs(1);
    double c = coeffs(2);
    double d = coeffs(3);
    double e = coeffs(4);

    double denom = 1 + d * d + e * e;
    double mole = (1 + d * d) * a + (1 + e * e) * b - 4 * a * b * c;

    meanCurvature = mole / (std::pow(denom, 1.5));

}


rbfParam calculaterbf(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr& x_cloud,
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr& refpoints,
    int paramenter                   //      0: luma;     1: Chroma;      2: Curv;
) {

    // 点云归一化
    PointType min_origin_point, max_origin_point;
    pcl::getMinMax3D(*refpoints, min_origin_point, max_origin_point);
    float maxLength;
    float xLength = max_origin_point.x - min_origin_point.x;
    float yLength = max_origin_point.y - min_origin_point.y;
    float zLength = max_origin_point.z - min_origin_point.z;

    maxLength = xLength > yLength ? xLength : yLength;
    maxLength = maxLength > zLength ? maxLength : zLength;

    for (int i = 0; i < refpoints->points.size(); i++) {
        PointType point;
        point.x = (double)((double)refpoints->points[i].x - min_origin_point.x) / (double)maxLength * 1024.0;
        point.y = (double)((double)refpoints->points[i].y - min_origin_point.y) / (double)maxLength * 1024.0;
        point.z = (double)((double)refpoints->points[i].z - min_origin_point.z) / (double)maxLength * 1024.0;
        refpoints->points[i].x = point.x;
        refpoints->points[i].y = point.y;
        refpoints->points[i].z = point.z;
    }

    for (int i = 0; i < x_cloud->points.size(); i++) {
        PointType point;
        point.x = (double)((double)x_cloud->points[i].x - min_origin_point.x) / (double)maxLength * 1024.0;
        point.y = (double)((double)x_cloud->points[i].y - min_origin_point.y) / (double)maxLength * 1024.0;
        point.z = (double)((double)x_cloud->points[i].z - min_origin_point.z) / (double)maxLength * 1024.0;
        x_cloud->points[i].x = point.x;
        x_cloud->points[i].y = point.y;
        x_cloud->points[i].z = point.z;
    }

    RecPointCloud<float> x_points;
    x_points.points.resize(x_cloud->points.size());

    for (int i = 0; i < x_cloud->points.size(); i++) {
        x_points.points[i].x = x_cloud->points[i].x;
        x_points.points[i].y = x_cloud->points[i].y;
        x_points.points[i].z = x_cloud->points[i].z;
    }

    KDTree reconPointIndex(3 /* dimensions */, x_points, KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
    reconPointIndex.buildIndex();

    std::vector<double> rbfA;
    rbfA.resize(refpoints->points.size());
    std::vector<double> rbfB;
    rbfB.resize(refpoints->points.size());
    std::vector<double> rbfC;
    rbfC.resize(refpoints->points.size());
    std::vector<double> rbfD;
    rbfD.resize(refpoints->points.size());
    std::vector<double> rbfOmega;
    rbfOmega.resize(refpoints->points.size());


    for (int j = 0; j < refpoints->points.size(); j++) {
        PointType point;
        point = refpoints->points[j];
        std::vector<float> target_point = { point.x, point.y, point.z };

        const size_t K = 30; 
        std::vector<uint32_t> indices(K);
        std::vector<float> distances(K);
        const size_t num_results = reconPointIndex.knnSearch(&target_point[0], K, &indices[0], &distances[0]);

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr rbf_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        for (int i = 0; i < K; i++) {
            rbf_cloud->push_back(x_cloud->points[indices[i]]);
        }
        std::vector<double> Curv;
        Curv.resize(K);
        if (paramenter == 2) {
            for (int i = 0; i < K; i++) {
                PointType pointCurv;
                pointCurv = rbf_cloud->points[i];
                std::vector<float> target_point = { pointCurv.x, pointCurv.y, pointCurv.z };
                const size_t T = 10; 
                std::vector<uint32_t> indicesCurv(T);
                std::vector<float> distancesCurv(T);
                const size_t num_results = reconPointIndex.knnSearch(&target_point[0], T, &indicesCurv[0], &distancesCurv[0]);
                double meanCurvature;
                fitSurfaceAndCalculateCurvature(x_cloud, indicesCurv, Curv[i]);
            }
        }

        //  *****************************************    RBF    ******************************************

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr filteredCloud(new pcl::PointCloud<pcl::PointXYZRGB>); 

        filteredCloud = rbf_cloud;

        const int quantificationPointNum = filteredCloud->size();

        Eigen::MatrixXd quantifiA1((quantificationPointNum + 4), (quantificationPointNum + 4));     
        Eigen::VectorXd quantifiCyuv(quantificationPointNum + 4);                             

        for (int i = 0; i < (quantificationPointNum + 4); i++) {
            for (int j = 0; j < (quantificationPointNum + 4); j++) {
                if ((i < quantificationPointNum) && (j < quantificationPointNum)) {

                    double A1x = (double)filteredCloud->points[i].x - filteredCloud->points[j].x;
                    double A1y = (double)filteredCloud->points[i].y - filteredCloud->points[j].y;
                    double A1z = (double)filteredCloud->points[i].z - filteredCloud->points[j].z;
                    double r = A1x * A1x + A1y * A1y + A1z * A1z;
                    quantifiA1(i, j) = f(r);
                    //std::cout << "A1(i, j) =  " << A1(i, j) << std::endl;
                }
                if ((i >= quantificationPointNum) && (j < quantificationPointNum)) {
                    if (i == quantificationPointNum) {
                        quantifiA1(i, j) = (double)filteredCloud->points[j].x;
                    }
                    if (i == (quantificationPointNum + 1)) {
                        quantifiA1(i, j) = (double)filteredCloud->points[j].y;
                    }
                    if (i == (quantificationPointNum + 2)) {
                        quantifiA1(i, j) = (double)filteredCloud->points[j].z;
                    }
                    if (i == (quantificationPointNum + 3)) {
                        quantifiA1(i, j) = 1;
                    }
                }
                else if ((i < quantificationPointNum) && (j >= quantificationPointNum)) {
                    if (j == quantificationPointNum) {
                        quantifiA1(i, j) = (double)filteredCloud->points[i].x;
                    }
                    if (j == (quantificationPointNum + 1)) {
                        quantifiA1(i, j) = (double)filteredCloud->points[i].y;
                    }
                    if (j == (quantificationPointNum + 2)) {
                        quantifiA1(i, j) = (double)filteredCloud->points[i].z;
                    }
                    if (j == (quantificationPointNum + 3)) {
                        quantifiA1(i, j) = 1;
                    }
                }
                else if ((i < quantificationPointNum) && (j >= quantificationPointNum)) {
                    if (j == quantificationPointNum) {
                        quantifiA1(i, j) = (double)filteredCloud->points[i].x;
                    }
                    if (j == (quantificationPointNum + 1)) {
                        quantifiA1(i, j) = (double)filteredCloud->points[i].y;
                    }
                    if (j == (quantificationPointNum + 2)) {
                        quantifiA1(i, j) = (double)filteredCloud->points[i].z;
                    }
                    if (j == (quantificationPointNum + 3)) {
                        quantifiA1(i, j) = 1;
                    }
                }
                else if ((i >= quantificationPointNum) && (j >= quantificationPointNum)) {
                    quantifiA1(i, j) = 0;
                }

            }
            if (i < quantificationPointNum) {
                if (paramenter == 0) {
                    quantifiCyuv(i) = (double)filteredCloud->points[i].r * 0.21 + (double)filteredCloud->points[i].g * 0.72 + (double)filteredCloud->points[i].b * 0.07;
                }
                if (paramenter == 1) {
                    float pointu = float(((double)filteredCloud->points[i].r * (-0.1146) + (double)filteredCloud->points[i].g * (-0.3854) + (double)filteredCloud->points[i].b * 0.5) + 128);
                    float pointv = float(((double)filteredCloud->points[i].r * 0.5 + (double)filteredCloud->points[i].g * (-0.4542) + (double)filteredCloud->points[i].b * (-0.0458)) + 128);
                    quantifiCyuv(i) = (pointu + pointv) / 2;
                }
                if (paramenter == 2) {
                    quantifiCyuv(i) = Curv[i];
                }

            }
            else
            {
                quantifiCyuv(i) = 0;
            }

        }

        Eigen::VectorXd quantifiXyuv;  //   
        quantifiXyuv = quantifiA1.householderQr().solve(quantifiCyuv);
        if (!quantifiXyuv.allFinite()) {
        }

        double meanOmega = 0;
        for (int i = 0; i < K; i++) {
            meanOmega += fabs(quantifiXyuv(i));
        }
        meanOmega /= K;

        rbfA[j] = quantifiXyuv(K);
        rbfB[j] = quantifiXyuv(K + 1);
        rbfC[j] = quantifiXyuv(K + 2);
        rbfD[j] = quantifiXyuv(K + 3);
        rbfOmega[j] = meanOmega;
    }
    return rbfParam(rbfA, rbfB, rbfC, rbfD, rbfOmega);
}

std::vector<std::string> splitString(const std::string& input, char delimiter) {
    std::vector<std::string> tokens;
    std::istringstream iss(input);
    std::string token;
    while (std::getline(iss, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

void writeToCSV(
    const std::vector<double>& HrbfA,
    const std::vector<double>& HrbfB,
    const std::vector<double>& HrbfC,
    const std::vector<double>& HrbfD,
    const std::vector<double>& HrbfOmega,
    const std::vector<double>& MrbfA,
    const std::vector<double>& MrbfB,
    const std::vector<double>& MrbfC,
    const std::vector<double>& MrbfD,
    const std::vector<double>& MrbfOmega,
    const std::vector<double>& LrbfA,
    const std::vector<double>& LrbfB,
    const std::vector<double>& LrbfC,
    const std::vector<double>& LrbfD,
    const std::vector<double>& LrbfOmega,
    const std::string& filename) {

    size_t size = HrbfA.size();

    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Error: Unable to open file." << std::endl;
        return;
    }

    // 写入 CSV 头
    file << "HrbfA,HrbfB,HrbfC,HrbfD,HrbfOmega,MrbfA,MrbfB,MrbfC,MrbfD,MrbfOmega,LrbfA,LrbfB,LrbfC,LrbfD,LrbfOmega\n";

    // 写入数据
    for (size_t i = 0; i < size; ++i) {
        file << HrbfA[i] << ','
            << HrbfB[i] << ','
            << HrbfC[i] << ','
            << HrbfD[i] << ','
            << HrbfOmega[i] << ','

            << MrbfA[i] << ','
            << MrbfB[i] << ','
            << MrbfC[i] << ','
            << MrbfD[i] << ','
            << MrbfOmega[i] << ','

            << LrbfA[i] << ','
            << LrbfB[i] << ','
            << LrbfC[i] << ','
            << LrbfD[i] << ','
            << LrbfOmega[i]
            << '\n';
    }

    file.close();
}


int main(int argc, char** argv)
{

    /*
    if (argc != 3) {
        std::cout << "Param amount is illegal ! Expected amount: 2, actual amount: " << argc << std::endl;
        return 0;
    }

    std::string originDataPath = argv[1];
    std::string testDataPath = argv[2];
    */

    std::string testDataPath = "G://study//data//2024_test_data//NWPU//loot_vox10_1200//loot_Normal//31.ply";
    std::string originDataPath = "G://study//code//202412020_Multi-scale_metric//point_QM//sampled//oringal.ply";

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr disCloud = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr refCloud = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();

    if (pcl::io::loadPLYFile<pcl::PointXYZRGB>(testDataPath, *disCloud) == -1)
    {
        PCL_ERROR("Couldn't read the PLY file.\n");
        return -1;
    }
    if (pcl::io::loadPLYFile<pcl::PointXYZRGB>(originDataPath, *refCloud) == -1)
    {
        PCL_ERROR("Couldn't read the PLY file.\n");
        return -1;
    }

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr h_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr m_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr l_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

    pcl::VoxelGrid<pcl::PointXYZRGB> sor;  
    sor.setInputCloud(disCloud);

    sor.setLeafSize(2.0f, 2.0f, 2.0f);
    sor.filter(*h_cloud);

    sor.setLeafSize(4.0f, 4.0f, 4.0f);
    sor.filter(*m_cloud);

    sor.setLeafSize(8.0f, 8.0f, 8.0f);
    sor.filter(*l_cloud);

   //  pcl::io::savePLYFile("cloud_filtered.ply", *h_cloud);

   // pcl::io::savePLYFileASCII("sampled_output.ply", *refpoints);

    rbfParam Hrbf = calculaterbf(h_cloud, refCloud, 1);
    rbfParam Mrbf = calculaterbf(m_cloud, refCloud, 1);
    rbfParam Lrbf = calculaterbf(l_cloud, refCloud, 1);
    //  std::cout << "rbf H:  " << Hrbf.rbfA[0] << std::endl;

    writeToCSV(Hrbf.rbfA, Hrbf.rbfB, Hrbf.rbfC, Hrbf.rbfD, Hrbf.rbfOmega, Mrbf.rbfA, Mrbf.rbfB, Mrbf.rbfC, Mrbf.rbfD, Mrbf.rbfOmega, Lrbf.rbfA, Lrbf.rbfB, Lrbf.rbfC, Lrbf.rbfD, Lrbf.rbfOmega, "output.csv");

    return 0;
}
