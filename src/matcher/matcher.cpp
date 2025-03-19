#include "matcher.hpp"



double AngleLinePlane(cv::Vec4d pi, cv::Vec3d v) {
  cv::Vec3d n(pi(0), pi(1), pi(2));

  double num = abs(v(0) * n(0) + v(1) * n(1) + v(2) * n(2));
  double den1 = sqrt(v(0) * v(0) + v(1) * v(1) + v(2) * v(2));
  double den2 = sqrt(n(0) * n(0) + n(1) * n(1) + n(2) * n(2));

  double beta = acos(num / (den1 * den2));
  double alpha = (M_PI / 2) - beta;
  return alpha;
}


double AngleLinePlane(cv::Vec4d pi, cv::Point3d v) {
  return AngleLinePlane(pi, cv::Vec3d(v.x, v.y, v.z));
}


float DistancePointLine(const cv::Point2f point, const cv::Vec3f &line) {
  return std::fabs(line(0) * point.x + line(1) * point.y + line(2)) / std::sqrt(line(0) * line(0) + line(1) * line(1));
}


cv::Vec4f EquationPlane(cv::Point3f p1, cv::Point3f p2, cv::Point3f p3) {
  float a1 = p2.x - p1.x;
  float b1 = p2.y - p1.y;
  float c1 = p2.z - p1.z;
  float a2 = p3.x - p1.x;
  float b2 = p3.y - p1.y;
  float c2 = p3.z - p1.z;
  float a = b1 * c2 - b2 * c1;
  float b = a2 * c1 - a1 * c2;
  float c = a1 * b2 - b1 * a2;
  float d = (-a * p1.x - b * p1.y - c * p1.z);

  cv::Vec4f piVec(a, b, c, d);
  return piVec;
}


cv::Vec4f EquationPlane(cv::Vec3f p1, cv::Vec3f p2, cv::Vec3f p3) {
  float a1 = p2(0) - p1(0);
  float b1 = p2(1) - p1(1);
  float c1 = p2(2) - p1(2);
  float a2 = p3(0) - p1(0);
  float b2 = p3(1) - p1(1);
  float c2 = p3(2) - p1(2);
  float a = b1 * c2 - b2 * c1;
  float b = a2 * c1 - a1 * c2;
  float c = a1 * b2 - b1 * a2;
  float d = (-a * p1(0) - b * p1(1) - c * p1(2));

  cv::Vec4f piVec(a, b, c, d);
  return piVec;
}


cv::Vec3f EquationLine(cv::Point2f p1, cv::Point2f p2) {
  float dx = p2.x - p1.x;
  float dy = p2.y - p1.y;
  float m = dy / dx;
  float c = p1.y - m * p1.x;

  cv::Vec3f eqLine(m, -1, c);
  return eqLine;
}


std::vector<double> flatten(std::vector<std::vector<double>> &v) {
  std::vector<double> flat;
  for (size_t i = 0; i < v.size(); i++)
    for (size_t j = 0; j < v[i].size(); j++)
      flat.push_back(v[i][j]);
  return flat;
}


void indices_from_flatten_position(int &i, int &j, int pos, int cols) {
  i = pos / cols;
  j = pos % cols;
}


std::vector<int> ordered_indices(const std::vector<double> &v) {
  std::vector<int> index(v.size());
  std::iota(index.begin(), index.end(), 0);
  std::sort(index.begin(), index.end(),
            [&v](int i, int j) { return v[i] < v[j]; });
  return index;
}


Matcher::Matcher() {}


void Matcher::MatchAngle(FisheyeLens* lens, Image* im1, Image* im2, double th) {
  cv::Point3f c1g = im1->cg();
  cv::Point3f c2g = im2->cg();
  
  for (auto kp_a : im1->kps_) {
    cv::Point2f pt_a = kp_a.pt;
    std::vector<double> pt_a_cil_cam = lens->Compute3D(pt_a.x, pt_a.y, false);
    cv::Point3d pt_a_cam = lens->CilToCart(pt_a_cil_cam[0], pt_a_cil_cam[1]);
    cv::Point3d pt_a_w = im1->PointGlobal(pt_a_cam);
    cv::Vec4f pi_a = EquationPlane(pt_a_w, c1g, c2g);
  
    std::vector<cv::Point3f> candidates_i;
    std::vector<cv::Point3f> candidates_crosscheck_i;
    std::vector<double> candidates_angle_i;
    std::vector<double> candidates_angle_crosscheck_i;

    for (auto kp_b: im2->kps_) {
      cv::Point2f pt_b = kp_b.pt;
      std::vector<double> pt_b_cil_cam = lens->Compute3D(pt_b.x, pt_b.y, false);
      cv::Point3f pt_b_cam = lens->CilToCart(pt_b_cil_cam[0], pt_b_cil_cam[1]);
      cv::Point3f pt_b_w = im2->PointGlobalRotation(pt_b_cam);

      // Check if point is under angle threshold
      double angle_a = AngleLinePlane(pi_a, pt_b_w);
      if (angle_a < th) {
        pt_b_w = im2->PointGlobalTranslation(pt_b_w);
        candidates_i.push_back(pt_b_w);
        candidates_angle_i.push_back(angle_a);
        

        cv::Vec4f pi_b = EquationPlane(pt_b_w, c1g, c2g);
        double angle_b = AngleLinePlane(pi_b, pt_a_w);
        if (angle_b < th) {
          candidates_crosscheck_i.push_back(pt_b_w);
          candidates_angle_crosscheck_i.push_back(sqrt(pow(angle_a,2) + pow(angle_b,2)));
        } else {
          candidates_angle_crosscheck_i.push_back(-1.0);
        }
      } else {
        candidates_angle_i.push_back(-1.0);
        candidates_angle_crosscheck_i.push_back(-1.0);
      }
    } 

    candidates_.push_back(candidates_i);
    candidates_crosscheck_.push_back(candidates_crosscheck_i);

    candidates_val_.push_back(candidates_angle_i);
    candidates_val_crosscheck_.push_back(candidates_angle_crosscheck_i);
  }
}


void Matcher::MatchSampson(FisheyeLens* lens, Image* im1, Image* im2, cv::Mat F, double th) {

  std::vector<cv::Point2f> kpoints1, kpoints2;
  for (size_t i = 0; i < im1->kps_.size(); i++)
    kpoints1.push_back(im1->kps_[i].pt);
  for (size_t i = 0; i < im2->kps_.size(); i++)
    kpoints2.push_back(im2->kps_[i].pt);

  std::vector<cv::Vec3f> gmlines1, gmlines2;
  cv::computeCorrespondEpilines(kpoints1, 1, F, gmlines1);
  cv::computeCorrespondEpilines(kpoints2, 2, F, gmlines2);
  
  double lx = 2*lens->cx();
  double ly = 2*lens->cy();

  for (size_t i = 0; i < kpoints1.size(); i++) {
    cv::Point3f pt0(0, -gmlines1[i][2] / gmlines1[i][1], 0.);
    cv::Point3f pt1(lx, -(gmlines1[i][2] + gmlines1[i][0] * lx) / gmlines1[i][1], 0.);
    cv::Vec3f line = EquationLine(cv::Point2f(pt0.x, pt0.y), cv::Point2f(pt1.x, pt1.y));

    cv::Mat point1m(cv::Point3d(kpoints1[i].x, kpoints1[i].y, 1.0));

    std::vector<cv::Point3f> candidates_i, candidates_crosscheck_i;
    std::vector<double> candidates_val_i, candidates_val_crosscheck_i;

    for (size_t j = 0; j < kpoints2.size(); j++) {
      cv::Point2f kp(kpoints2[j].x, kpoints2[j].y);

      cv::Mat point2m(cv::Point3d(kpoints2[j].x, kpoints2[j].y, 1.0));

      if (cv::sampsonDistance(point1m, point2m, F) <= th) {
        double dist_l2 = norm(im1->desc_.row(i), im2->desc_.row(j), cv::NORM_L2);
        candidates_i.push_back(cv::Point3f(kpoints2[j].x, kpoints2[j].y, 1.0));
        candidates_crosscheck_i.push_back(cv::Point3f(kpoints2[j].x, kpoints2[j].y, 1.0));
        candidates_val_i.push_back(dist_l2);
        candidates_val_crosscheck_i.push_back(dist_l2);
      } else {
        candidates_val_i.push_back(0.0);
        candidates_val_crosscheck_i.push_back(0.0);
      }
    }


    candidates_.push_back(candidates_i);
    candidates_crosscheck_.push_back(candidates_crosscheck_i);
    candidates_val_.push_back(candidates_val_i);
    candidates_val_crosscheck_.push_back(candidates_val_crosscheck_i);

  }
}



std::vector<cv::Point2f> Matcher::SampsonRegion(FisheyeLens* lens, Image* im1, Image* im2, cv::Mat F, int pt_idx, double th){
  // Draw epipolar line and search region
  std::vector<cv::Point2f> kpoints1, kpoints2;
  for (size_t i = 0; i < im1->kps_.size(); i++)
    kpoints1.push_back(im1->kps_[i].pt);
  for (size_t i = 0; i < im2->kps_.size(); i++)
    kpoints2.push_back(im2->kps_[i].pt);

  std::vector<cv::Vec3f> gmlines1, gmlines2;
  cv::computeCorrespondEpilines(kpoints1, 1, F, gmlines1);
  cv::computeCorrespondEpilines(kpoints2, 2, F, gmlines2);
  
  double lx = 2*lens->cx();
  double ly = 2*lens->cy();

  // Epipolar line
  cv::Point3f pt_l_0(0, -gmlines1[pt_idx][2] / gmlines1[pt_idx][1], 0.);
  cv::Point3f pt_l_1(lx, -(gmlines1[pt_idx][2] + gmlines1[pt_idx][0] * lx) / gmlines1[pt_idx][1], 0.);
  cv::Vec3f line = EquationLine(cv::Point2f(pt_l_0.x, pt_l_0.y), cv::Point2f(pt_l_1.x, pt_l_1.y));

  std::vector<cv::Point2f> points_epipolar_th;
  for (auto pt: im2->points_) {
    float dist = DistancePointLine(pt, line);
    if (dist < th) {
      points_epipolar_th.push_back(pt);
    }
  }

  return points_epipolar_th;
}




std::vector<cv::DMatch> Matcher::NNMatches(double th) {
  std::vector<cv::DMatch> nn;

  for (size_t i = 0; i < candidates_val_.size(); i++) {
    int idx_i = -1;
    double dist_i = 1000000.;

    for (size_t j = 0; j < candidates_val_[i].size(); j++) {
      if (candidates_val_[i][j] >= 0. && candidates_val_[i][j] < dist_i){

        double dist_i_tmp = candidates_val_[i][j];

        int idx_j = -1;
        double dist_j = -1000000.;

        for (size_t k = 0; k < candidates_val_.size(); k++)        {
          if (candidates_val_[k][j] >= 0. && candidates_val_[k][j] < dist_j && candidates_val_[k][j] < dist_i_tmp) {
            idx_j = k;
            dist_j = candidates_val_[k][j];
          }
        }

        if (idx_j < 0) {
          idx_i = j;
          dist_i = dist_i_tmp;
        }
      }
    }

    if (dist_i >= 0. && dist_i <= th) {
      cv::DMatch dm;
      dm.queryIdx = i;
      dm.trainIdx = idx_i;
      dm.imgIdx = 0;
      dm.distance = dist_i;
      nn.push_back(dm);

      for (size_t j = 0; j < candidates_val_.size(); j++) {
        candidates_val_[j][idx_i] = -1.;
      }
    }
  }

  return nn;
}
  

std::vector<cv::DMatch> Matcher::DescMatches(Image* im1, Image* im2, double th) {
  std::vector<std::vector<double>> sift_descriptor_simmilarity;

  for (size_t i = 0; i < candidates_val_crosscheck_.size(); i++) {
    std::vector<double> row;
    for (size_t j = 0; j < candidates_val_crosscheck_[i].size(); j++) {
      if (candidates_val_crosscheck_[i][j] >= 0.) {
        double dist_l2 = norm(im1->desc_.row(i), im2->desc_.row(j), cv::NORM_L2);
        row.push_back(dist_l2);
      } else {
        row.push_back(-1.);
      }
    }
    sift_descriptor_simmilarity.push_back(row);
  }

  std::vector<cv::DMatch> nn;

  std::vector<double> flat = flatten(sift_descriptor_simmilarity);
  std::vector<int> index = ordered_indices(flat);

  for (size_t i = 0; i < index.size(); i++) {
    int idx_i, idx_j;
    indices_from_flatten_position(idx_i, idx_j, index[i], sift_descriptor_simmilarity[0].size());
    double dist_ij = sift_descriptor_simmilarity[idx_i][idx_j];
    if (dist_ij > 0.0 && dist_ij < th) {
      nn.push_back(cv::DMatch(idx_i, idx_j, dist_ij));
      for (size_t k = 0; k < sift_descriptor_simmilarity.size(); k++)
        sift_descriptor_simmilarity[k][idx_j] = -1.;
      for (size_t k = 0; k < sift_descriptor_simmilarity[idx_i].size(); k++)
        sift_descriptor_simmilarity[idx_i][k] = -1.;
    }
  }

  return nn;
}


cv::Point2f Matcher::Match2D(Image* im, 
                             FisheyeLens* lens, 
                             std::vector<cv::DMatch> candidates, 
                             int point_id) {
  std::vector<cv::Point3f> points_candidates;
  cv::Point2f kp_cand_2d(0,0);
  int match_nn = -1;
  for (auto match : candidates){
    if (match.queryIdx == point_id){
      match_nn = match.trainIdx;
    }
  }
  if (match_nn > 0) {
    kp_cand_2d = im->kps_[match_nn].pt;
  }
  return kp_cand_2d;
}


std::vector<cv::Point3f> Matcher::Match3D(Image* im, 
                                          FisheyeLens* lens, 
                                          std::vector<cv::DMatch> candidates, 
                                          int point_id) {
  std::vector<cv::Point3f> points_candidates;
  int match_nn = -1;
  for (auto match : candidates){
    if (match.queryIdx == point_id){
      match_nn = match.trainIdx;
    }
  }
  if (match_nn > 0) {
    cv::Point2f kp_cand_2d = im->kps_[match_nn].pt;
    std::vector<double> kp_cand_3d = lens->Compute3D(kp_cand_2d.x, kp_cand_2d.y, false);
    cv::Point3d kp_cand_3d_xyz_img = lens->CilToCart(kp_cand_3d[0], kp_cand_3d[1]);
    cv::Point3d kp_cand_3d_xyz = im->PointGlobal(kp_cand_3d_xyz_img);
    points_candidates.push_back(kp_cand_3d_xyz);
  }
  return points_candidates;
}