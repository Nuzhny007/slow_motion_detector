#include <iostream>
#include <chrono>

#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>

#include <mdetection/BaseDetector.h>
#include <mtracking/BaseTracker.h>

///----------------------------------------------------------------------
void DrawData(cv::Mat frame, const std::vector<TrackingObject>& tracks, int framesCounter, int currTime)
{
	std::cout << "Frame (" << framesCounter << "): tracks = " << tracks.size() << ", time = " << currTime << std::endl;

    static std::vector<cv::Scalar> colors;
    if (colors.empty())
    {
        colors.emplace_back(255, 0, 0);
        colors.emplace_back(0, 255, 0);
        colors.emplace_back(0, 0, 255);
        colors.emplace_back(255, 255, 0);
        colors.emplace_back(0, 255, 255);
        colors.emplace_back(255, 0, 255);
        colors.emplace_back(255, 127, 255);
        colors.emplace_back(127, 0, 255);
        colors.emplace_back(127, 0, 127);
    }

	for (size_t i = 0; i < tracks.size(); ++i)
	{
		const auto& track = tracks[i];

        std::cout << "track: rrect [" << track.m_rrect.size << " from " << track.m_rrect.center << ", " << track.m_rrect.angle << "]" << std::endl;

		if (track.IsRobust(2,             // Minimal trajectory size
			0.1f,                         // Minimal ratio raw_trajectory_points / trajectory_lenght
			cv::Size2f(0.2f, 5.0f),       // Min and max ratio: width / height
			1))
		{
			cv::Scalar color = cv::Scalar(0, 255, 0);
			cv::Point2f rectPoints[4];
			track.m_rrect.points(rectPoints);
			for (int i = 0; i < 4; ++i)
			{
				cv::line(frame, rectPoints[i], rectPoints[(i + 1) % 4], color);
			}

			cv::Scalar cl = colors[track.m_ID.ID2Module(colors.size())];

			for (size_t j = 0; j < track.m_trace.size() - 1; ++j)
			{
				const TrajectoryPoint& pt1 = track.m_trace.at(j);
				const TrajectoryPoint& pt2 = track.m_trace.at(j + 1);
				cv::line(frame, pt1.m_prediction, pt2.m_prediction, cl, 1, cv::LINE_AA);
				if (!pt2.m_hasRaw)
					cv::circle(frame, pt2.m_prediction, 4, cl, 1, cv::LINE_AA);
			}

			cv::Rect brect = track.m_rrect.boundingRect();
			std::stringstream label;
			label << track.m_ID.ID2Str();
			if (track.m_type != bad_type)
				label << ": " << TypeConverter::Type2Str(track.m_type);
			if (track.m_confidence > 0)
				label << ", " << std::fixed << std::setw(2) << std::setprecision(2) << track.m_confidence;

			int baseLine = 0;
			double fontScale = (frame.cols < 1920) ? 0.5 : 0.7;
			cv::Size labelSize = cv::getTextSize(label.str(), cv::FONT_HERSHEY_TRIPLEX, fontScale, 1, &baseLine);
			if (brect.x < 0)
			{
				brect.width = std::min(brect.width, frame.cols - 1);
				brect.x = 0;
			}
			else if (brect.x + brect.width >= frame.cols)
			{
				brect.x = std::max(0, frame.cols - brect.width - 1);
				brect.width = std::min(brect.width, frame.cols - 1);
			}
			if (brect.y - labelSize.height < 0)
			{
				brect.height = std::min(brect.height, frame.rows - 1);
				brect.y = labelSize.height;
			}
			else if (brect.y + brect.height >= frame.rows)
			{
				brect.y = std::max(0, frame.rows - brect.height - 1);
				brect.height = std::min(brect.height, frame.rows - 1);
			}
			DrawFilledRect(frame, cv::Rect(cv::Point(brect.x, brect.y - labelSize.height), cv::Size(labelSize.width, labelSize.height + baseLine)), cv::Scalar(200, 200, 200), 150);
			cv::putText(frame, label.str(), brect.tl(), cv::FONT_HERSHEY_TRIPLEX, fontScale, cv::Scalar(0, 0, 0));
		}
	}
}

///----------------------------------------------------------------------
int main(int argc, char** argv)
{
    const char* keys =
    {
        "{ @1                  |../data/atrium.avi  | movie file | }"
        "{ out                 |                    | Name of result video file | }"
        "{ g gpu               |0                   | Use OpenCL acceleration | }"
    };

    cv::CommandLineParser parser(argc, argv, keys);

    parser.printMessage();

    bool useOCL = parser.get<int>("gpu") != 0;
    cv::ocl::setUseOpenCL(useOCL);
    std::cout << (cv::ocl::useOpenCL() ? "OpenCL is enabled" : "OpenCL not used") << std::endl;

    std::string inFile = parser.get<std::string>(0);
    std::string outFile = parser.get<std::string>("out");

    cv::VideoCapture capture(inFile);
    if (!capture.isOpened())
    {
        std::cerr << "Video " << inFile << " open error!" << std::endl;
        return -1;
    }

    cv::Mat frame;
    capture >> frame;
    if (frame.empty())
    {
        std::cerr << "Frame is empty" << std::endl;
        return -2;
    }

    double fps = capture.get(cv::CAP_PROP_FPS);

    // Create detector
    config_t detector_config;
    detector_config.emplace("useRotatedRect", "0");
    detector_config.emplace("history", std::to_string(cvRound(5000 * fps)));
    detector_config.emplace("nmixtures", "3");
    detector_config.emplace("backgroundRatio", "0.7");
    detector_config.emplace("noiseSigma", "0");
    std::unique_ptr<BaseDetector> detector = BaseDetector::CreateDetector(tracking::Detectors::Motion_MOG, detector_config, frame.getUMat(cv::ACCESS_READ));
    detector->SetMinObjectSize(cv::Size(4, 4));

    // Create tracker
    TrackerSettings trackerSettings;
    trackerSettings.m_tracker = tracking::UniversalTracker;
    trackerSettings.SetDistance(tracking::DistJaccard);
    trackerSettings.m_kalmanType = tracking::KalmanLinear;
    trackerSettings.m_filterGoal = tracking::FilterCenter;
    trackerSettings.m_lostTrackType = tracking::TrackNone; // Use visual objects tracker for collisions resolving. Used if m_filterGoal == tracking::FilterRect
    trackerSettings.m_matchType = tracking::MatchLAPJV;
    trackerSettings.m_useAcceleration = false;             // Use constant acceleration motion model
    trackerSettings.m_dt = trackerSettings.m_useAcceleration ? 0.05f : 0.3f; // Delta time for Kalman filter
    trackerSettings.m_accelNoiseMag = 0.1f;                // Accel noise magnitude for Kalman filter
    trackerSettings.m_distThres = 0.95f;                   // Distance threshold between region and object on two frames
    trackerSettings.m_minAreaRadiusPix = frame.rows / 10.f;
    trackerSettings.m_minAreaRadiusK = 0.8f;
    trackerSettings.m_useAbandonedDetection = false;
    trackerSettings.m_maximumAllowedLostTime = 10.;       // Maximum allowed lost time
    trackerSettings.m_maxTraceLength = 10.;               // Maximum trace length
    std::unique_ptr<BaseTracker> tracker = BaseTracker::CreateTracker(trackerSettings, fps);
    

    cv::VideoWriter writer;


    time_point_t startTimeStamp = std::chrono::system_clock::now();
    double freq = cv::getTickFrequency();

    double readPeriodSeconds = 2.;
    int readPeriodFrames = cvRound(readPeriodSeconds * fps);

    for (;;)
    {
        capture >> frame;
        int framesCounter = cvRound(capture.get(cv::CAP_PROP_POS_FRAMES));
        if (frame.empty())
        {
            std::cerr << "Frame " << framesCounter << " is empty" << std::endl;
            break;
        }
        capture.set(cv::CAP_PROP_POS_FRAMES, framesCounter + readPeriodFrames);

        std::chrono::time_point<std::chrono::system_clock> currTime = startTimeStamp + std::chrono::milliseconds(cvRound(framesCounter * (1000 / fps)));

        int64 t1 = cv::getTickCount();

        cv::UMat um = frame.getUMat(cv::ACCESS_READ);
        detector->Detect(um);
        auto regions = detector->GetDetects();
        
        std::vector<TrackingObject> tracks;
        tracker->Update(regions, um, currTime);
        tracker->GetTracks(tracks);

        int64 t2 = cv::getTickCount();
        int workTime = cvRound(1000 * (t2 - t1) / freq);

        DrawData(frame, tracks, framesCounter, workTime);
        detector->CalcMotionMap(frame);

        cv::imshow("Video", frame);
        int k = cv::waitKey(0);
        if (k == 27)
            break;

        if (!outFile.empty())
        {
            if (!writer.isOpened())
            {
                writer.open(outFile, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, frame.size(), true);
            }
            if (writer.isOpened())
                writer << frame;
        }
    }

    return 0;
}
