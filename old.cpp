
    cv::Ptr<cv::DescriptorMatcher> matcher_1 = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    std::vector< std::vector<cv::DMatch> > knn_matches_1;

    std::vector<std::vector<double> > distances;
    for (size_t i=0; i<match_candidates.size(); i++){
        std::vector<double> distances_i;
        for (size_t j=0; j<match_candidates[i].size(); j++){
            double dist_l2  = norm(desc1.row(i),desc2.row(match_candidates[i][j]),cv::NORM_L2);
            distances_i.push_back(dist_l2);
        }
        distances.push_back(distances_i);
    }

    // For each kp2 to which kp1s has been marked as candidate
    std::vector<std::vector<int> > inv_index;
    std::vector<std::vector<double> > inv_distances;
    for (size_t i=0; i<kps2.size(); i++){
        std::vector<int> inv_index_i;
        std::vector<double> inv_distances_i;
        for (size_t j=0; j<match_candidates.size(); j++){
            for (size_t k=0; k<match_candidates[j].size(); k++){
                if (match_candidates[j][k]==i){
                    inv_index_i.push_back(j);
                    inv_distances_i.push_back(distances[j][k]);
                }
            }
        }
        inv_index.push_back(inv_index_i);
        inv_distances.push_back(inv_distances_i);
    }


    std::vector<cv::DMatch> egmatches;
    for (size_t i=0; i<inv_distances.size(); i++){
        double score = -1.0;
        int id_1 = -1;
        int id_2 = -1;
        if (inv_distances[i].size() > 0){
            for (size_t j=0; j<inv_distances[i].size(); j++){
                if (inv_distances[i][j] > score && inv_distances[i][j] > -1.0){
                    score = inv_distances[i][j];
                    id_1 = inv_index[i][j];
                    id_2 = j;
                }
            }
        }
        if (id_1 > -1){
            for (size_t j=0; j<inv_index.size(); j++){
                for (size_t k=0; k<inv_index[j].size(); k++){
                    if (inv_index[j][k]==id_1){
                        inv_index[j][k] = -1;
                        inv_distances[j][k] = -1.0;
                    }
                }
            }

            cv::DMatch dm;
            dm.queryIdx = id_1;
            dm.trainIdx = id_2;
            dm.imgIdx = 0;
            for (size_t j=0; j<match_candidates[id_1].size(); j++){
                if (match_candidates[id_1][j] == id_2){
                    dm.distance = (float) distances[id_1][j];
                    break;
                }
            }
            egmatches.push_back(dm);
        }
    }


    std::vector<cv::DMatch> egmatches1;
    double th = 1000.0;
    for (size_t i=0; i<egmatches.size(); i++){
        if (egmatches[i].distance <= th){
            egmatches1.push_back(egmatches[i]);
        }
    }

    std::cout << "good / matches / kps2 = " << egmatches1.size() << " / " << egmatches.size() << " / " << kps2.size() << std::endl;

    cv::Mat imout;
    cv::drawMatches(im1,kps1,im2,kps2,egmatches1,imout);
    resize_and_display("eg",imout,0.5);
