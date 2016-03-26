# -*- coding: utf-8 -*-
"""
@author: benjamin lefaudeux

Find the inter frame motion, try to make it robust to mismatches
"""

import cv2 as cv
import numpy as np
import time


class TrackImage:

    def __init__(self):
        self.n_max_corners = 400
        self.corners_q_level = 4
        self.reset = False
        self.reset_ratio = 0.3

    def compensate_interframe_motion(self, ref_frame, new_frame, method='shi_tomasi'):
        # Testing different methods here to align the frames

        if method == 'shi_tomasi':
            transform, success = self.__motion_estimation_shi_tomasi(ref_frame, new_frame)

        elif method == 'orb':
            transform, success = self.__motion_estimation_orb(ref_frame, new_frame)

        elif method == 'sift':
            #    acc_frame_aligned = self.__compensate_SIFT( ref_frame, new_frame)
            print "Cannot use SIFT right now..."
            transform, success = None, False

        else:
            ValueError('Wrong argument for motion compensation')

        if success:
            return cv.warpPerspective(new_frame, transform, new_frame.shape[2::-1]), True

        return None, False

    def __motion_estimation_orb(self, ref_frame, new_frame):

        # Create an ORB detector
        detector = cv.FastFeatureDetector(16, True)
        # detector = cv2.GridAdaptedFeatureDetector(detector)
        extractor = cv.DescriptorExtractor_create('ORB')

        # Test with ORB corners :
        _min_match_count = 20

        # find the keypoints and descriptors with ORB
        kp1 = detector.detect(new_frame)
        k1, des1 = extractor.compute(new_frame, kp1)

        kp2 = detector.detect(ref_frame)
        k2, des2 = extractor.compute(ref_frame, kp2)

        # Match using bruteforce
        matcher = cv.DescriptorMatcher_create('BruteForce-Hamming')
        matches = matcher.match(des1, des2)

        # keep only the reasonable matches
        dist = [m.distance for m in matches]        # store all the good matches as per Lowe's ratio test.
        thres_dist = (sum(dist) / len(dist)) * 0.5  # threshold: half the mean
        good_matches = [m for m in matches if m.distance < thres_dist]

        # - bring the second picture in the reference referential
        if len(good_matches) > _min_match_count:
            print "Enough matchs for compensation - %d/%d" % (len(good_matches), _min_match_count)
            self.corners = np.float32([k1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            self.corners_next = np.float32([k2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            transform, mask = cv.findHomography(self.corners, self.corners_next, cv.RANSAC, 5.0)

            # Check that the transform indeed explains the corners shifts ?
            mask_match = [m for m in mask if m == 1]
            match_ratio = len(mask_match) / float(len(mask))
            if match_ratio < self.reset_ratio:
                self.reset = True
                print "Accumulation reset, track lost - %d" % match_ratio
                return False

            # Align the previous accumulated frame
            return transform, True

        else:
            print "Not enough matches are found - %d/%d" % (len(good_matches), _min_match_count)
            return None, False

    def __motion_estimation_shi_tomasi(self, ref_frame, new_frame):
        """
        Measure and compensate for inter-frame motion:
        - get points on both frames
        -- we use Shi & Tomasi here, to be adapted ?
        @rtype : opencv frame
        """
        corners = cv.goodFeaturesToTrack(ref_frame, self.n_max_corners, .01, 50)

        # - track points
        corners_next, status, _ = cv.calcOpticalFlowPyrLK(ref_frame, new_frame, corners)

        # - track back (more reliable)
        corners_next_back, status_back, _ = cv.calcOpticalFlowPyrLK(new_frame, ref_frame, corners_next)

        # - sort out to keep reliable points :
        corners, corners_next = self.__sort_corners(corners, corners_next, status, corners_next_back, status_back)

        # - compute the transformation from the tracked pattern
        # -- estimate the rigid transform
        transform, mask = cv.findHomography(corners, corners_next, cv.RANSAC, 5.0)

        # -- see if this transform explains most of the displacements (thresholded..)
        if len(mask[mask > 0]) > 20:  # TODO: More robust test here ?
            print "Enough match for motion compensation"
            return transform, True

        else:
            print "Not finding enough matchs - {}".format(len(mask[mask > 0]))
            return None, False

    @staticmethod
    def __motion_estimation_sift(ref_frame, new_frame):
        # Test with SIFT corners :
        _min_match_count = 10
        _flann_index_kdtree = 0

        # Initiate SIFT detector
        sift = cv.SIFT()

        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(ref_frame, None)
        kp2, des2 = sift.detectAndCompute(new_frame, None)

        index_params = dict(algorithm=_flann_index_kdtree, trees=5)
        search_params = dict(checks=50)

        flann = cv.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1, des2, k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

        # - bring the second picture in the current referential
        if len(good) > _min_match_count:
            print "Enough matches for compensation - %d/%d" % (len(good), _min_match_count)
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            transform, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()

            h, w = ref_frame.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            pts_transform = cv.perspectiveTransform(pts, transform)
            #        new_frame = cv2.polylines(new_frame,[np.int32(transform)],True,255,3, cv2.LINE_AA)

        else:
            print "Not enough matches are found - %d/%d" % (len(good), _min_match_count)
            matchesMask = None

    @staticmethod
    def __draw_vec(img, corners, corners_next):
        """
        Draw motion vectors on the picture

        @param img: picture to draw onto
        @param corners: initial keypoints position
        @param corners_next: position after tracking
        """
        try:
            corn_xy = corners.reshape((-1, 2))
            corn_xy_next = corners_next.reshape((-1, 2))

            i = 0
            for x, y in corn_xy:
                cv.line(img, (int(x), int(y)), (int(corn_xy_next[i, 0]), int(corn_xy_next[i, 1])), [0, 0, 255], 5)
                i += 1

        except ValueError:
            print "Problem printing the motion vectors"

    @staticmethod
    def __sort_corners(corners_init, corners_tracked, status_tracked,
                       corners_tracked_back, status_tracked_back, max_dist=0.5):

        # Check that the status value is 1, and that
        i = 0
        nice_points = []
        for c1 in corners_init:
            c2 = corners_tracked_back[i]
            dist = cv.norm(c1, c2)

            if status_tracked[i] and status_tracked_back[i] and dist < max_dist:
                nice_points.append(i)

            i += 1

        return [corners_init[nice_points], corners_tracked[nice_points]]

    @property
    def show(self):
        keep_going = False

        # Show the current combined picture
        print "Showing frame {}".format(self.n_fused_frames)

        # Do all the resizing beforehand
        frame_fusion_resize = cv.resize(self.frame_acc_disp, (800, 600))

        # Onscreen print
        cv.putText(frame_fusion_resize, "Space continues, Esc leaves \n R resets",
                   (30, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, 255)

        cv.namedWindow("FrameFusion")
        cv.imshow("FrameFusion", frame_fusion_resize)
        cv.waitKey(5)

        # Show the initial picture
        cv.namedWindow('Raw frame')

        # - Show tracked features
        self.__draw_vec(self.frame_prev, self.corners, self.corners_next)

        frame_raw_resize = cv.resize(self.frame_prev, (800, 600))
        cv.imshow('Raw frame', frame_raw_resize)
        cv.waitKey(5)

        start_time = time.time()

        while 1:
            k = cv.waitKey(33)

            current_time = time.time()

            # Escape quits
            if 27 == k or 1048603 == k:
                keep_going = False
                cv.destroyWindow('FrameFusion')
                cv.destroyWindow('Raw frame')
                break

            # Space continues
            elif 32 == k or 1048608 == k:
                keep_going = True
                break

            # R resets the accumulation
            elif ord('r') == k or 1048690 == k:
                keep_going = True
                self.reset = True
                print "Reset the accumulation"
                break

            # Timer went through, time to leave
            elif (current_time - start_time) > 1:
                keep_going = True
                print "Waited enough, next frame !"
                break

            elif k != -1:
                print k

        return keep_going


