
/*
    void cv::fisheye::distortPoints	(	InputArray 	undistorted,
        OutputArray 	distorted,
        InputArray 	K,
        InputArray 	D,
        double 	alpha = 0 
    )	

    void cv::fisheye::projectPoints	(	InputArray 	objectPoints,
        OutputArray 	imagePoints,
        InputArray 	rvec,
        InputArray 	tvec,
        InputArray 	K,
        InputArray 	D,
        double 	alpha = 0,
        OutputArray 	jacobian = noArray() 
    )		

    void cv::fisheye::undistortImage	(	InputArray 	distorted,
        OutputArray 	undistorted,
        InputArray 	K,
        InputArray 	D,
        InputArray 	Knew = cv::noArray(),
        const Size & 	new_size = Size() 
    )		

    void cv::fisheye::undistortPoints	(	InputArray 	distorted,
        OutputArray 	undistorted,
        InputArray 	K,
        InputArray 	D,
        InputArray 	R = noArray(),
        InputArray 	P = noArray() 
    )	
*/