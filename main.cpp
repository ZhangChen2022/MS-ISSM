#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <Eigen/Dense>
#include <nanoflann.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <algorithm>

double f(double a) {
    return (std::exp(-0.5 * a));
}

// nanoflann 
template <typename T>
struct RecPointCloud {
    struct Point { T x, y, z; };
    std::vector<Point> points;
    inline size_t kdtree_get_point_count() const { return points.size(); }
    inline T kdtree_get_pt(const size_t idx, const size_t dim) const {
        if (dim == 0) return points[idx].x;
        else if (dim == 1) return points[idx].y;
        else return points[idx].z;
    }
    template <class BBOX> bool kdtree_get_bbox(BBOX& /* bb */) const { return false; }
};

typedef nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<double, RecPointCloud<double>>,
    RecPointCloud<double>, 3, size_t> KDTree;

struct rbfParam {
    std::vector<double> rbfA, rbfB, rbfC, rbfD, rbfOmega;
};

void fitSurfaceAndCalculateCurvature(const RecPointCloud<double>& cloud, const std::vector<size_t>& indices, double& meanCurvature) {
    int num_points = indices.size();
    if (num_points < 6) { meanCurvature = 0; return; }

    Eigen::MatrixXd A(num_points, 6);
    Eigen::VectorXd B(num_points);
    for (int i = 0; i < num_points; ++i) {
        double x = cloud.points[indices[i]].x;
        double y = cloud.points[indices[i]].y;
        double z = cloud.points[indices[i]].z;
        A(i, 0) = x * x; A(i, 1) = y * y; A(i, 2) = x * y;
        A(i, 3) = x; A(i, 4) = y; A(i, 5) = 1;
        B(i) = z;
    }
    Eigen::VectorXd coeffs = A.colPivHouseholderQr().solve(B);
    double a = coeffs(0), b = coeffs(1), c = coeffs(2), d = coeffs(3), e = coeffs(4);
    double denom = 1 + d * d + e * e;
    double mole = (1 + d * d) * a + (1 + e * e) * b - 4 * a * b * c;
    meanCurvature = mole / std::pow(denom, 1.5);
}

rbfParam calculaterbf(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& x_cloud,
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& refpoints,
    int paramenter
) {
    size_t ref_size = refpoints->points.size();
    size_t x_size = x_cloud->points.size();

    RecPointCloud<double> local_ref, local_x;
    local_ref.points.resize(ref_size);
    local_x.points.resize(x_size);

    for (size_t i = 0; i < ref_size; i++) {
        local_ref.points[i].x = refpoints->points[i].x;
        local_ref.points[i].y = refpoints->points[i].y;
        local_ref.points[i].z = refpoints->points[i].z;
    }
    for (size_t i = 0; i < x_size; i++) {
        local_x.points[i].x = x_cloud->points[i].x;
        local_x.points[i].y = x_cloud->points[i].y;
        local_x.points[i].z = x_cloud->points[i].z;
    }

    KDTree reconPointIndex(3, local_x, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    reconPointIndex.buildIndex();

    std::vector<double> rbfA(ref_size), rbfB(ref_size), rbfC(ref_size), rbfD(ref_size), rbfOmega(ref_size);
    const size_t K = std::min((size_t)30, x_size);

    for (size_t j = 0; j < ref_size; j++) {
        double query_pt[3] = { local_ref.points[j].x, local_ref.points[j].y, local_ref.points[j].z };
        std::vector<size_t> indices(K);
        std::vector<double> dists_sq(K);
        reconPointIndex.knnSearch(&query_pt[0], K, &indices[0], &dists_sq[0]);

        std::vector<double> CurvValues(K, 0.0);
        if (paramenter == 2) {
            for (size_t i = 0; i < K; i++) {
                double sub_query[3] = { local_x.points[indices[i]].x, local_x.points[indices[i]].y, local_x.points[indices[i]].z };
                const size_t T = 10;
                std::vector<size_t> sub_indices(T);
                std::vector<double> sub_dists(T);
                reconPointIndex.knnSearch(&sub_query[0], T, &sub_indices[0], &sub_dists[0]);
                fitSurfaceAndCalculateCurvature(local_x, sub_indices, CurvValues[i]);
            }
        }

        Eigen::MatrixXd matA = Eigen::MatrixXd::Zero(K + 4, K + 4);
        Eigen::VectorXd vecB = Eigen::VectorXd::Zero(K + 4);

        for (size_t row = 0; row < K; ++row) {
            for (size_t col = 0; col < K; ++col) {
                double r2 = std::pow(local_x.points[indices[row]].x - local_x.points[indices[col]].x, 2) +
                    std::pow(local_x.points[indices[row]].y - local_x.points[indices[col]].y, 2) +
                    std::pow(local_x.points[indices[row]].z - local_x.points[indices[col]].z, 2);
                matA(row, col) = f(r2);
            }
            matA(row, K) = local_x.points[indices[row]].x;
            matA(row, K + 1) = local_x.points[indices[row]].y;
            matA(row, K + 2) = local_x.points[indices[row]].z;
            matA(row, K + 3) = 1.0;
            matA(K, row) = local_x.points[indices[row]].x;
            matA(K + 1, row) = local_x.points[indices[row]].y;
            matA(K + 2, row) = local_x.points[indices[row]].z;
            matA(K + 3, row) = 1.0;

            const auto& p = x_cloud->points[indices[row]];
            if (paramenter == 0) vecB(row) = p.r * 0.2126 + p.g * 0.7152 + p.b * 0.0722;
            else if (paramenter == 1) {
                double u = -0.1146 * p.r - 0.3854 * p.g + 0.5 * p.b + 128;
                double v = 0.5 * p.r - 0.4542 * p.g - 0.0458 * p.b + 128;
                vecB(row) = (u + v) / 2.0;
            }
            else vecB(row) = CurvValues[row];
        }

        Eigen::VectorXd sol = matA.colPivHouseholderQr().solve(vecB);
        double omega = 0;
        for (int i = 0; i < K; i++) omega += std::abs(sol(i));

        rbfA[j] = sol(K); rbfB[j] = sol(K + 1); rbfC[j] = sol(K + 2); rbfD[j] = sol(K + 3);
        rbfOmega[j] = omega / K;
    }
    return { rbfA, rbfB, rbfC, rbfD, rbfOmega };
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <target_cloud.ply> <reference_cloud.ply> <output.csv>" << std::endl;
        return -1;
    }

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr disCloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr orgCloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::io::loadPLYFile(argv[1], *disCloud);
    pcl::io::loadPLYFile(argv[2], *orgCloud);

    Eigen::Vector4f min_pt_v4, max_pt_v4;
    pcl::getMinMax3D(*orgCloud, min_pt_v4, max_pt_v4);
    Eigen::Vector3f base_min = min_pt_v4.head<3>();
    float max_len = (max_pt_v4.head<3>() - base_min).maxCoeff();
    if (max_len < 1e-6f) max_len = 1.0f; 

    for (auto& p : orgCloud->points) {
        p.x = (p.x - base_min.x()) / max_len * 1024.0f;
        p.y = (p.y - base_min.y()) / max_len * 1024.0f;
        p.z = (p.z - base_min.z()) / max_len * 1024.0f;
    }
    for (auto& p : disCloud->points) {
        p.x = (p.x - base_min.x()) / max_len * 1024.0f;
        p.y = (p.y - base_min.y()) / max_len * 1024.0f;
        p.z = (p.z - base_min.z()) / max_len * 1024.0f;
    }

    auto start = std::chrono::high_resolution_clock::now();

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr h_cloud(new pcl::PointCloud<pcl::PointXYZRGB>),
        m_cloud(new pcl::PointCloud<pcl::PointXYZRGB>),
        l_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::VoxelGrid<pcl::PointXYZRGB> sor;

    sor.setInputCloud(disCloud);
    sor.setLeafSize(2.0f, 2.0f, 2.0f); sor.filter(*h_cloud);
    sor.setLeafSize(4.0f, 4.0f, 4.0f); sor.filter(*m_cloud);
    sor.setLeafSize(8.0f, 8.0f, 8.0f); sor.filter(*l_cloud);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr refpoints(new pcl::PointCloud<pcl::PointXYZRGB>);
    sor.setInputCloud(orgCloud);
    sor.setLeafSize(32.0f, 32.0f, 32.0f);
    sor.filter(*refpoints);

    rbfParam LH = calculaterbf(h_cloud, refpoints, 0);
    rbfParam LM = calculaterbf(m_cloud, refpoints, 0);
    rbfParam LL = calculaterbf(l_cloud, refpoints, 0);
    rbfParam CH = calculaterbf(h_cloud, refpoints, 1);
    rbfParam CM = calculaterbf(m_cloud, refpoints, 1);
    rbfParam CL = calculaterbf(l_cloud, refpoints, 1);
    rbfParam VH = calculaterbf(h_cloud, refpoints, 2);
    rbfParam VM = calculaterbf(m_cloud, refpoints, 2);
    rbfParam VL = calculaterbf(l_cloud, refpoints, 2);

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Algorithm Time: " << std::chrono::duration<double>(end - start).count() << "s" << std::endl;

    std::ofstream outfile(argv[3]);
    outfile << "LH_A,LH_B,LH_C,LH_D,LH_O,LM_A,LM_B,LM_C,LM_D,LM_O,LL_A,LL_B,LL_C,LL_D,LL_O,"
        << "CH_A,CH_B,CH_C,CH_D,CH_O,CM_A,CM_B,CM_C,CM_D,CM_O,CL_A,CL_B,CL_C,CL_D,CL_O,"
        << "VH_A,VH_B,VH_C,VH_D,VH_O,VM_A,VM_B,VM_C,VM_D,VM_O,VL_A,VL_B,VL_C,VL_D,VL_O" << std::endl;

    for (size_t i = 0; i < refpoints->points.size(); ++i) {
        outfile << LH.rbfA[i] << "," << LH.rbfB[i] << "," << LH.rbfC[i] << "," << LH.rbfD[i] << "," << LH.rbfOmega[i] << ","
            << LM.rbfA[i] << "," << LM.rbfB[i] << "," << LM.rbfC[i] << "," << LM.rbfD[i] << "," << LM.rbfOmega[i] << ","
            << LL.rbfA[i] << "," << LL.rbfB[i] << "," << LL.rbfC[i] << "," << LL.rbfD[i] << "," << LL.rbfOmega[i] << ","
            << CH.rbfA[i] << "," << CH.rbfB[i] << "," << CH.rbfC[i] << "," << CH.rbfD[i] << "," << CH.rbfOmega[i] << ","
            << CM.rbfA[i] << "," << CM.rbfB[i] << "," << CM.rbfC[i] << "," << CM.rbfD[i] << "," << CM.rbfOmega[i] << ","
            << CL.rbfA[i] << "," << CL.rbfB[i] << "," << CL.rbfC[i] << "," << CL.rbfD[i] << "," << CL.rbfOmega[i] << ","
            << VH.rbfA[i] << "," << VH.rbfB[i] << "," << VH.rbfC[i] << "," << VH.rbfD[i] << "," << VH.rbfOmega[i] << ","
            << VM.rbfA[i] << "," << VM.rbfB[i] << "," << VM.rbfC[i] << "," << VM.rbfD[i] << "," << VM.rbfOmega[i] << ","
            << VL.rbfA[i] << "," << VL.rbfB[i] << "," << VL.rbfC[i] << "," << VL.rbfD[i] << "," << VL.rbfOmega[i] << std::endl;
    }
    return 0;
}