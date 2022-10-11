import numpy as np

np.random.seed(42)


if __name__ == "__main__":

    max_number_of_tracks = 232
    max_number_of_tracks_power_2 = 256
    max_number_of_tracks_log_2 = 8
    batch_size = 50
    
    eps = 0.15
    verbose = False
    save_intermediate = False
    from batched_dbscan import BatchedDBSCAN

    for i in range(10):
        file_i = i
        #z0_file = f"/media/lucas/QS/binaries-trk/OldKF_TTbar_170K_quality-{file_i}-trk-z0.bin"
        #pt_file = f"/media/lucas/QS/binaries-trk/OldKF_TTbar_170K_quality-{file_i}-trk-pt.bin"
        store = "/home/kirby/data/binaries-trk-100/"
        z0_file = store + f"b-{file_i}-trk-z0.bin"
        pt_file = store + f"b-{file_i}-trk-pt.bin"
        z0 = np.fromfile(z0_file, dtype=np.float32)
        pt = np.fromfile(pt_file, dtype=np.float32)


        db = BatchedDBSCAN(z0, pt, eps, batch_size, max_number_of_tracks, verbose, save_intermediate)

        db.fit()


        # print(db.boundaries_batches[0])
        print(db.z0_pv, db.max_pt)
        
        # import json
        # with open('merged_list.json', 'w') as f:
        #     json.dump(db.merged_list, f, indent=4)
    
