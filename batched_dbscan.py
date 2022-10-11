import numpy as np
import math
import pandas as pd
import itertools
import copy

from sklearn.cluster import DBSCAN

np.random.seed(42)

np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)


class BatchedDBSCAN:
    def __init__(
        self,
        z0,
        pt,
        eps,
        batch_size,
        max_number_of_tracks,
        verbose: bool = False,
        save_intermediate: bool = False,
    ):

        # predefined constants
        self.z0_boundary = 21
        self.pt_boundary = 0
        self.minPts = 2

        self.eps = eps
        self.batch_size = batch_size
        self.max_number_of_tracks = int(max_number_of_tracks)
        self.n_batches = math.ceil(self.max_number_of_tracks / self.batch_size)

        # Max number of tracks including all batches
        self.max_n_tracks_batched = self.batch_size * self.n_batches
        self.max_n_clusters_batch = math.ceil(self.batch_size / self.minPts) #
        self.max_n_clusters = math.ceil(self.max_n_tracks_batched / self.minPts)

        # need to pad vectors to have the fixed size
        n_pad = self.max_number_of_tracks - z0.shape[0]
        self.z0 = self.pad_vector(z0, n_pad, self.z0_boundary)
        self.pt = self.pad_vector(pt, n_pad, self.pt_boundary)

        # Prefix sum variables
        self.max_number_of_tracks_power_2 = (
            1 << (self.max_number_of_tracks - 1).bit_length()
        )
        self.batch_size_power_2 = 1 << (self.batch_size - 1).bit_length()
        self.max_number_of_tracks_log_2 = np.log2(self.max_number_of_tracks_power_2)
        self.batch_size_log_2 = np.log2(self.batch_size_power_2)

        # Dictionaries for recording the results
        self.results = {}
        self.results_sklearn = {}
        self.merged_list = []

        # Debug options
        self.verbose = verbose
        self.save_intermediate = save_intermediate

        # Cluster array indices
        # pT, z0_low, z0_high, Noise
        self.pt_idx = 0
        self.z0_low_idx = 1
        self.z0_high_idx = 2
        self.noise_idx = 3

    def print_state(self):
        
        print(f"Max number of tracks: {self.max_number_of_tracks}")
        print(f"batch size: {self.batch_size}")
        print(f"n_batches: {self.n_batches}")
        print(f"max number of tracks batched: {self.max_n_tracks_batched}")
        print(f"max number of clusters per batch: {self.max_n_clusters_batch}")
        print(f"max number of clusters: {self.max_n_clusters}")
        print(f"z0 shape: {self.z0.shape[0]}")
        print(f"max_number of tracks rounded to the nearest power of 2: {self.max_number_of_tracks_power_2}")
        print(f"max number of tracks log base2: {self.max_number_of_tracks_log_2}")
        print(f"batch size to the nearest power of 2: {self.batch_size_power_2}")
        print(f"batch size to log base2: {self.batch_size_log_2}")
        

    
    def pad_vector(self, vec: np.array, n_pad: int, value: int):
        """pads vector to a set size with given value"""

        vec_to_pad = value * np.ones(n_pad)
        vec = np.append(vec, vec_to_pad)

        return vec

    def build_tracks(self, z0, pt):
        """builds tracks batches"""

        # Shape is determined by the size of the batch, z0, pt and label (not used atm)
        track_batch = np.zeros((self.batch_size, 3))
        track_batch[:, 0] = z0
        track_batch[:, 1] = pt

        # sort the tracks by z0
        track_batch = track_batch[track_batch[:, 0].argsort()]

        return track_batch

    def prefix_sum(self, arr):
        """
        Calculates the prefix sum of pT.
        Warning, requires array to be of size thats log base of 2.
        """
        size_log2 = int(np.log2(arr.shape[0]))

        # up-sweep
        for d in range(0, size_log2, 1):
            step_size = 2**d
            double_step_size = step_size * 2

            for i in range(0, arr.shape[0], double_step_size):
                arr[i + double_step_size - 1] += arr[i + step_size - 1]

        # down-sweep
        arr[arr.shape[0] - 1] = 0
        d = size_log2 - 1

        while d >= 0:
            step_size = 2**d
            double_step_size = step_size * 2
            for i in range(0, arr.shape[0], double_step_size):
                tmp = arr[i + step_size - 1]
                arr[i + step_size - 1] = arr[i + double_step_size - 1]
                arr[i + double_step_size - 1] += tmp
            d -= 1

        return arr

    def find_left_boundaries(self, tracks):

        left_boundaries = np.zeros(self.batch_size, dtype=bool)

        # first value is always a left boundary
        left_boundaries[0] = 1

        for i in range(1, self.batch_size):
            _t = tracks[i]

            if _t[0] - tracks[i - 1][0] > self.eps:
                tracks[i][2] = -1
                left_boundaries[i] = 1
            else:
                left_boundaries[i] = 0

        self.left_boundaries = left_boundaries
        return left_boundaries

    def find_right_boundaries(self, left_boundaries, rs, tracks):

        max_tracks = self.batch_size

        boundaries = np.zeros((max_tracks, 7))
        is_noise = np.ones((max_tracks, 1))

        for i in range(max_tracks - 1):

            left_edge = left_boundaries[i] and not (left_boundaries[i + 1])  # 1, 0
            right_edge = not (left_boundaries[i]) and left_boundaries[i + 1]  # 0, 1
            check_noise = (left_boundaries[i] == 1) and (left_boundaries[i + 1] == 1)

            if left_edge or right_edge:
                boundaries[i][0] = i
                boundaries[i][1] = rs[i]
                boundaries[i][2] = rs[i + 1]
                boundaries[i][3] = rs[i + 1] - rs[i]
                boundaries[i][4] = tracks[i, 0]
                boundaries[i][5] = tracks[i + 1, 0]
                is_noise[i] = 0
            elif check_noise:
                boundaries[i][0] = i
                boundaries[i][1] = rs[i]
                boundaries[i][2] = rs[i + 1]
                boundaries[i][3] = rs[i + 1] - rs[i]
                boundaries[i][4] = tracks[i, 0]
                boundaries[i][5] = tracks[i, 0]
                boundaries[i][6] = 1
                # is_noise[i] = 1
            else:
                boundaries[i][0] = max_tracks
                boundaries[i][1] = 0
                boundaries[i][2] = 0
                boundaries[i][3] = 0
                boundaries[i][4] = 21
                boundaries[i][5] = 21
                is_noise[i] = 0

        # Check for the last boundary

        
        if left_boundaries[max_tracks - 1] and is_noise[max_tracks - 1] == 0:
            boundaries[max_tracks - 1][0] = max_tracks
            boundaries[max_tracks - 1][1] = 0
            boundaries[max_tracks - 1][2] = 0
            boundaries[max_tracks - 1][3] = 0
            boundaries[max_tracks - 1][4] = 21
            boundaries[max_tracks - 1][5] = 21
        elif left_boundaries[max_tracks - 1] and is_noise[max_tracks - 1] == 1:

            boundaries[max_tracks - 1][0] = max_tracks - 1
            boundaries[max_tracks - 1][1] = rs[max_tracks - 1]
            boundaries[max_tracks - 1][2] = rs[max_tracks]
            boundaries[max_tracks - 1][3] = rs[max_tracks] - rs[max_tracks - 1]
            boundaries[max_tracks - 1][4] = tracks[max_tracks - 1, 0]
            boundaries[max_tracks - 1][5] = tracks[max_tracks - 1, 0]
            boundaries[max_tracks - 1][6] = 1
        else:
            boundaries[max_tracks - 1][0] = max_tracks - 1
            boundaries[max_tracks - 1][1] = rs[max_tracks - 1]
            boundaries[max_tracks - 1][2] = rs[max_tracks]
            boundaries[max_tracks - 1][3] = rs[max_tracks] - rs[max_tracks - 1]
            boundaries[max_tracks - 1][4] = tracks[max_tracks - 1, 0]
            boundaries[max_tracks - 1][5] = tracks[max_tracks - 1, 0]

        # boundaries[:, 6] = is_noise
        # Sort boundaries by the index
        sort_idx = boundaries[:, 0].argsort()
        boundaries = boundaries[sort_idx]
        is_noise = is_noise[sort_idx]
        self.is_noise = is_noise
        return boundaries

    def get_vertex(self, cluster_of_tracks: np.array) -> float:
        """
        Calculates the median z0 of the cluster of tracks
        """

        n_size = cluster_of_tracks.shape[0]

        if n_size % 2 == 0:
            return 0.5 * (
                cluster_of_tracks[n_size // 2] + cluster_of_tracks[n_size // 2 - 1]
            )
        else:
            return cluster_of_tracks[n_size // 2]

    def initialize_clusters(self, max_n_clusters: int) -> np.array:

        clusters = np.zeros((max_n_clusters, 4))

        clusters[:, self.z0_low_idx] = 21
        clusters[:, self.z0_high_idx] = 21

        return clusters

    def convert_boundaries_to_clusters(self, boundaries: np.array):
        bound_i = 0
        cluster_j = 0
        n_boundaries = boundaries.shape[0]
        clusters = self.initialize_clusters(n_boundaries)
        for _ in range(n_boundaries):
            # if boundaries[bound_i, 4] == 21:
            #     break
            if (bound_i+2) >= n_boundaries:
                # print("breakpoint")
                break
            noise = boundaries[bound_i, -1]
            # print("bound_i: ", bound_i, "cluster_j: ", cluster_j, "noise: ", noise)

            if noise:
                z0_low = boundaries[bound_i, 4]
                z0_high = boundaries[bound_i, 5]
                pt_sum = boundaries[bound_i, 2] - boundaries[bound_i, 1]

                clusters[cluster_j, self.z0_low_idx] = z0_low
                clusters[cluster_j, self.z0_high_idx] = z0_high
                clusters[cluster_j, self.pt_idx] = pt_sum
                clusters[cluster_j, self.noise_idx] = noise
                bound_i += 1
                cluster_j += 1
            else:
                z0_low = boundaries[bound_i, 4]
                z0_high = boundaries[bound_i + 1, 4]
                pt_sum = boundaries[bound_i + 1, 2] - boundaries[bound_i, 1]

                clusters[cluster_j, self.z0_low_idx] = z0_low
                clusters[cluster_j, self.z0_high_idx] = z0_high
                clusters[cluster_j, self.pt_idx] = pt_sum
                clusters[cluster_j, self.noise_idx] = noise

                bound_i += 2
                cluster_j += 1
            # if bound_i >= n_boundaries:
            #     print("break")
            #     break

        return clusters

    def clusters_overlap(self, ci, cj) -> bool:
        # |---- c_i -----|   |---- c_j ----|
        case1 = ci[self.z0_low_idx] - self.eps <= cj[self.z0_high_idx]
        case2 = ci[self.z0_high_idx] + self.eps >= cj[self.z0_low_idx]

        overlap = case1 and case2

        return overlap

    def record_merging(self, ci, cj, cn):

        zi_low = round(ci[self.z0_low_idx], 2)
        zi_high = round(ci[self.z0_high_idx], 2)
        zi_pt = round(ci[self.pt_idx], 2)
        ci_str = f"[{zi_low}, {zi_high}, {zi_pt}]"

        zj_low = round(cj[self.z0_low_idx], 2)
        zj_high = round(cj[self.z0_high_idx], 2)
        zj_pt = round(cj[self.pt_idx], 2)
        cj_str = f"[{zj_low}, {zj_high}, {zj_pt}]"

        zn_low = round(cn[self.z0_low_idx], 2)
        zn_high = round(cn[self.z0_high_idx], 2)
        zn_pt = round(cn[self.pt_idx], 2)
        cn_str = f"[{zn_low}, {zn_high}, {zn_pt}]"

        merged_str = ci_str + " + " + cj_str + " -> " + cn_str
        self.merged_list.append(merged_str)

    def merge_clusters(self, c: np.array) -> np.array:

        clusters = c.copy()
        
        n_clusters = clusters.shape[0]
        if self.n_batches == 1:
            self.max_pt_i = np.argmax(clusters[:, self.pt_idx])
            self.max_pt = clusters[self.max_pt_i, self.pt_idx]
            self.merge_count = 0
            return clusters

        else:
            max_pt = 0
            max_pt_i = 0
            merge_count = 0

            comb = list(itertools.combinations(range(n_clusters), 2))
            self.comb = comb

            to_merge = 9 * np.ones((n_clusters, n_clusters))

            for i, j in comb:

                # skip if cluster  is outside detector
                if (clusters[i, self.z0_low_idx] >= 21) or (
                    clusters[j, self.z0_low_idx] >= 21
                ):
                    continue

                ci = copy.copy(clusters[i, :])
                cj = copy.copy(clusters[j, :])

                overlap = self.clusters_overlap(clusters[i, :], clusters[j, :])
                to_merge[i, j] = overlap
                if overlap:

                    # If cluster j is noise, then upon merging it is no-longer noise
                    cj_noise = clusters[j, self.noise_idx]

                    if cj_noise:
                        clusters[j, self.noise_idx] = 0

                    merge_count += 1

                    # Expand boundaries of cluster after merging
                    if clusters[i, self.z0_low_idx] < clusters[j, self.z0_low_idx]:
                        clusters[j, self.z0_low_idx] = clusters[i, self.z0_low_idx]
                    if clusters[i, self.z0_high_idx] > clusters[j, self.z0_high_idx]:
                        clusters[j, self.z0_high_idx] = clusters[i, self.z0_high_idx]

                    # Add the pT of the cluster being merged.
                    clusters[j, self.pt_idx] += clusters[i, self.pt_idx]

                    # Erase merged cluster.
                    clusters[i, self.pt_idx] = 0
                    clusters[i, self.z0_low_idx] = 21
                    clusters[i, self.z0_high_idx] = 21
                    clusters[i, self.noise_idx] = 0

                    self.record_merging(ci, cj, clusters[j, :])

                # check if the pT_sum max is now higher
                # Need to protect against selecting a noise point as PV
                if (max_pt < clusters[j, self.pt_idx]) and (
                    clusters[j, self.noise_idx] != 1
                ):
                    max_pt = clusters[j, self.pt_idx]
                    max_pt_i = j
            self.to_merge = pd.DataFrame(to_merge)
            self.max_pt = max_pt
            self.max_pt_i = max_pt_i
            self.merge_count = merge_count
            return pd.DataFrame(
                clusters, columns=["pt_sum", "z0_low", "z0_high", "noise"]
            )

    def fitsklearn(self):
        start_idx = 0
        end_idx = start_idx + self.batch_size
        n_pad = (self.n_batches * self.batch_size) - self.z0.shape[0]
        self.z0 = self.pad_vector(self.z0, n_pad, 21)
        self.pt = self.pad_vector(self.pt, n_pad, 0)

        clusters_df = pd.DataFrame({})
        clusters = self.initialize_clusters(self.max_n_tracks_batched)

        for i in range(self.n_batches):
            start_idx = i * self.batch_size
            end_idx = (i + 1) * self.batch_size
            z0_batch = self.z0[start_idx:end_idx]
            pt_batch = self.pt[start_idx:end_idx]

            _db = DBSCAN(eps=0.15, min_samples=2).fit(z0_batch.reshape(-1, 1))

            _results = pd.DataFrame(
                {"z0": z0_batch, "pt": pt_batch, "label": _db.labels_}
            )
            max_label = _results.label.max()
            n_noise = _results[_results.label == -1].shape[0]

            _results.loc[_results.label == -1, "label"] = (
                np.arange(n_noise) + max_label + 1
            )

            clusters_batch = _results.groupby(["label"]).agg(
                {"z0": [np.min, np.max], "pt": [np.sum, "count"]}
            )
            clusters_batch.columns = ["z0_low", "z0_high", "pt_sum", "ntracks"]
            clusters_batch["noise"] = 0
            clusters_batch.loc[clusters_batch["ntracks"] < 2, "noise"] = 1
            clusters_batch = clusters_batch.drop(columns=["ntracks"])

            self.results_sklearn[i] = clusters_batch

            clusters_df = pd.concat([clusters_df, clusters_batch])

        n_clusters = clusters_df.shape[0]

        clusters[0:n_clusters, self.pt_idx] = clusters_df["pt_sum"]
        clusters[0:n_clusters, self.z0_low_idx] = clusters_df["z0_low"]
        clusters[0:n_clusters, self.z0_high_idx] = clusters_df["z0_high"]
        clusters[0:n_clusters, self.noise_idx] = clusters_df["noise"]

        self.clusters_unmerged = pd.DataFrame(
            clusters.copy(), columns=["pt_sum", "z0_low", "z0_high", "noise"]
        )

        clusters_merged = self.merge_clusters(clusters)
        self.clusters_merged = pd.DataFrame(
            clusters_merged, columns=["pt_sum", "z0_low", "z0_high", "noise"]
        )

        pv_z0_low = self.clusters_merged.iloc[self.max_pt_i, self.z0_low_idx]
        pv_z0_high = self.clusters_merged.iloc[self.max_pt_i, self.z0_high_idx]

        z0_pv = np.median(self.z0[(self.z0 >= pv_z0_low) & (self.z0 <= pv_z0_high)])
        self.z0_pv_skl = z0_pv


    def fit(self):

        np.set_printoptions(precision=2)
        np.set_printoptions(suppress=True)
        
        if self.verbose:
            self.print_state()
        
        start_idx = 0
        end_idx = start_idx + self.batch_size
        # Need to pad vectors to match the size of n_batches*batch_size
        n_pad = (self.n_batches * self.batch_size) - self.z0.shape[0]
        self.z0 = self.pad_vector(self.z0, n_pad, 21)
        self.pt = self.pad_vector(self.pt, n_pad, 0)

        # clusters = self.initialize_clusters(self.max_n_clusters)
        clusters = self.initialize_clusters(self.max_n_tracks_batched)
        # print(self.max_n_tracks_batched)
        # print(self.batch_size, self.n_batches)
        # print(clusters.shape)

        self.z0_batches = {}
        self.pt_batches = {}
        self.rs_batches = {}
        self.left_boundaries_batches = {}
        self.boundaries_batches = {}
        self.clusters_batches = {}

        pv_cluster = np.zeros((1, 4))
        merge_count = 0
        for i in range(self.n_batches):
            # print(f"---------- batch number: {i} ---------")
            start_idx = i * self.batch_size
            end_idx = (i + 1) * self.batch_size
            
            # if i == 1:
            #     break

            z0_batch = self.z0[start_idx:end_idx]
            pt_batch = self.pt[start_idx:end_idx]

            track_batch = self.build_tracks(z0_batch, pt_batch)
            self.tracks = track_batch

            rs_batch = self.pad_vector(
                track_batch[:, 1], self.batch_size_power_2 - self.batch_size, 0
            )

            rs_batch = self.prefix_sum(rs_batch)
            self.rs = rs_batch

            # Storing batches
            self.z0_batches[i] = track_batch[:, 0]
            self.pt_batches[i] = track_batch[:, 1]
            self.rs_batches[i] = rs_batch

            # Finding Left Boundaries
            left_boundaries = self.find_left_boundaries(track_batch)
            self.left_boundaries_batches[i] = left_boundaries
            # print("left_boundaries shape: ", left_boundaries.shape)
            
            # Finding Right Boundaries
            np.save('left_boundaries.npy',left_boundaries)
            np.save('rs_batch.npy',rs_batch)
            np.save('track_batch.npy',track_batch)
            
            boundaries = self.find_right_boundaries(
                left_boundaries, rs_batch, track_batch
            )
            self.boundaries_batches[i] = boundaries
            # print("boundaries shape: ", boundaries.shape)
            
            self.boundaries = boundaries
            # np.save('boundaries.npy', boundaries)
            clusters_batch = self.convert_boundaries_to_clusters(boundaries)
            self.clusters_batches[i] = clusters_batch
       
            clusters[
                i * self.batch_size : (i + 1) * self.batch_size, :
            ] = clusters_batch

            self.results[i] = clusters_batch
        clusters = self.merge_clusters(clusters)

        self.clusters = clusters

        # Find pv_cluster
        # print(type(clusters))
        if self.n_batches == 1:
            pv_cluster[0, :] = clusters[self.max_pt_i, :]
        else:
            pv_cluster[0, :] = clusters.iloc[self.max_pt_i, :]

        # print(self.max_pt, self.max_pt_i)
        # print(f"Merged count: {self.merge_count}")

        pv_tracks = []

        for i in range(self.max_number_of_tracks):
            z0_trk = self.z0[i]

            if (z0_trk >= pv_cluster[0, self.z0_low_idx]) and (
                z0_trk <= pv_cluster[0, self.z0_high_idx]
            ):
                pv_tracks.append(z0_trk)

        median_vertex = self.get_vertex(np.array(pv_tracks))
        self.z0_pv = np.median(pv_tracks)

        # print(f"mean: {np.mean(pv_tracks)}")
        # print(f"median: {np.median(pv_tracks)}")
        # print(f"median2: {median_vertex}")