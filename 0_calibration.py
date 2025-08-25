import os
import cv2
import torch
import argparse
import numpy as np

class FanBeam:
    def __init__(self, angle_y, T, image_path, beads_2d):
        image = cv2.imread(image_path)
        self.H, self.W = image.shape[:2]
        self.Cx = self.W / 2 
        self.Cy = self.H / 2
        self.near = 0
        self.far = 1000
        self.image_path = image_path 
        self.beads_2d = beads_2d
        self.theta = angle_y
        self.R = self.get_rot_mat(self.theta) 
        self.T = T
        self.cam_center = - np.dot(self.R.T, self.T)
    
    def get_rot_mat(self, theta):
        theta_y = theta
        rot_y = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                          [0, 1, 0],
                          [-np.sin(theta_y), 0, np.cos(theta_y)]])
        return rot_y
    
    def project_beads(self, beads_3d, DSD):
        X = beads_3d[:, 0]
        Z = beads_3d[:, 2]
        Tx, Tz = self.T[0], self.T[2]
        theta = self.theta

        X_ = X * np.cos(theta) + Z * np.sin(theta) + Tx
        Z_ = Z * np.cos(theta) - X * np.sin(theta) + Tz

        u = self.Cx + (DSD * X_) / Z_
        v = self.Cy + (DSD * (beads_3d[:, 1])) / Z_

        return np.stack([u, v], axis=-1)

class FanbeamCalibrator: 
    def __init__(self, beads_3d_sets, beads_2d_sets, image_paths, num_cameras, calibration_sets, DSD, n_iter, lr=0.1, output_folder="output", optim_dsd=False, optim_beads=False, scheduler=None):
        self.beads_3d_sets = beads_3d_sets  
        self.beads_2d_sets = beads_2d_sets  
        self.image_paths = image_paths 
        self.num_cameras = num_cameras
        self.num_sets = len(beads_3d_sets)  
        self.DSD = torch.tensor(DSD, dtype=torch.float32, requires_grad=optim_dsd) 
        self.n_iter = n_iter
        self.lr = lr
        self.output_folder = output_folder
        self.calibration_sets = calibration_sets
        self.optim_dsd = optim_dsd
        self.optim_beads = optim_beads
        self.scheduler_type = scheduler

        self.params = torch.tensor([0., 1000, 0.], dtype=torch.float32).repeat(self.num_cameras, 1).clone().detach().requires_grad_(True)
        
        self.fanbeams = []
        for i in range(self.num_cameras):
            self.fanbeams.append(FanBeam(angle_y=0.0, T=np.array([0, 1000, 0]), image_path=self.image_paths[i], beads_2d=None))

        params_to_optimize = [self.params]
        if self.optim_dsd:
            params_to_optimize.append(self.DSD)
        if self.optim_beads:
            for beads_3d_set in self.beads_3d_sets:
                params_to_optimize.append(beads_3d_set)

        self.optimizer = torch.optim.Adam(params_to_optimize, lr=self.lr)

        if self.scheduler_type == "steplr":
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=2000, gamma=0.01)
        elif self.scheduler_type == "exponential":
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)
        else:
            self.scheduler = None

        self.losses = []

    def loss(self, params, beads_3d, beads_2d, camera_index):
        beads_3d_flipped = beads_3d.clone()
        beads_3d_flipped[:, 0] = -beads_3d_flipped[:, 0] 

        Tx, Tz, theta = params[camera_index]
        theta = theta * 0.01
        X = beads_3d_flipped[:, 0]
        Y = beads_3d_flipped[:, 2]
        Z = beads_3d_flipped[:, 1]

        self.fanbeams[camera_index].T = np.array([Tx.item(), 0, Tz.item()])
        self.fanbeams[camera_index].theta = theta.item()
        self.fanbeams[camera_index].R = self.fanbeams[camera_index].get_rot_mat(theta.item())

        Cx = self.fanbeams[camera_index].Cx 
        DSD = self.DSD
        X_ = X * torch.cos(theta) + Z * torch.sin(theta) + Tx
        Z_ = Z * torch.cos(theta) - X * torch.sin(theta) + Tz

        u = Cx + (DSD * X_) / Z_
        v = beads_2d[:,1]

        error = torch.nn.functional.mse_loss(u, beads_2d[:,0])
        return error
    
    def run(self):
        self.iter = 0 
        for i in range(self.n_iter):
            self.optimizer.zero_grad()
            total_loss = 0

            for set_index in range(self.num_sets):
                for camera_index in range(self.num_cameras):
                    beads_3d = self.beads_3d_sets[set_index]  
                    beads_2d = torch.tensor(self.beads_2d_sets[set_index][camera_index], dtype=torch.float32)

                    loss = self.loss(self.params, beads_3d, beads_2d, camera_index)
                    total_loss += loss

            total_loss = total_loss / (self.num_cameras * self.num_sets)
            total_loss.backward()
            self.optimizer.step()

            if self.scheduler:
                self.scheduler.step()

            self.losses.append(total_loss.item())
            print(f"Iteration {i+1}/{self.n_iter}, Loss: {total_loss.item()}")

    def save_params(self, output_path):
        params_np = self.params.detach().numpy()  
        DSD_np = self.DSD.item()  
        
        calibration_results = []

        for i in range(self.num_cameras):
            Tx, Tz, theta = params_np[i]
            calibration_results.append([Tx, Tz, theta, DSD_np]) 

        calibration_results = np.array(calibration_results)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, calibration_results)
        print(f"Saved calibration results to {output_path}")

        if self.optim_beads:
            for set_index, beads_3d_set in enumerate(self.beads_3d_sets):
                set_basename = os.path.basename(self.calibration_sets[set_index])
                beads_output_path = os.path.join(self.output_folder, f"optimized_beads_{set_basename}.npy")
                np.save(beads_output_path, beads_3d_set.detach().numpy())
                print(f"Saved optimized beads to {beads_output_path}")

def load_bead_data(calibration_sets, optim_beads=False):
    beads_2d_set = []
    beads_3d_set = []
    image_paths = []

    for set_dir in calibration_sets:
        set_basename = os.path.basename(set_dir)


        images = sorted([os.path.join(set_dir, f) for f in os.listdir(set_dir) if f.endswith('.png')])

        beads_2d = np.load(os.path.join(set_dir, set_basename+'_2d.npy'))
        beads_3d = np.load(os.path.join(set_dir, set_basename+'_3d.npy'))
        beads_3d = torch.tensor(beads_3d, dtype=torch.float32, requires_grad=optim_beads)
        
        beads_2d_set.append(beads_2d)
        beads_3d_set.append(beads_3d)
        image_paths.append(images)
    
    return beads_2d_set, beads_3d_set, image_paths, calibration_sets

def calibrate(input_dir, DSD_init, n_iter, output_file, optim_dsd, optim_beads, lr, scheduler):
    calibration_sets = []

    for folder in sorted(os.listdir(input_dir)):
        full_path = os.path.join(input_dir, folder)
        if os.path.isdir(full_path):
            calibration_sets.append(full_path)

    beads_2d_set, beads_3d_set, image_paths, calibration_sets = load_bead_data(calibration_sets, optim_beads)
    num_cameras = beads_2d_set[0].shape[0]

    print(f"Running calibration across all sets... DSD: {optim_dsd}, Beads: {optim_beads}, Scheduler: {scheduler}")
    calibrator = FanbeamCalibrator(beads_3d_set, beads_2d_set, image_paths[0], num_cameras, DSD=DSD_init, n_iter=n_iter, lr=lr, calibration_sets=calibration_sets, optim_dsd=optim_dsd, optim_beads=optim_beads, scheduler=scheduler)
    calibrator.run()
    calibrator.save_params(output_path=output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize fanbeam camera parameters and optionally optimize 3D bead locations using reprojection error.")
    parser.add_argument('--input', required=False, default="data/calibration", help="Calibration path containing sets of x-ray images.")
    parser.add_argument('--DSD', required=False, default=1100, help="Initial Distance from source to detector (DSD).")
    parser.add_argument("--lr", type=float, required=False, default=0.1, help="Learning rate for optimization.")
    parser.add_argument("--iter", type=int, required=False, default=10000, help="Number of optimization iterations.")
    parser.add_argument("--output", required=False, default="output/calibration_results.npy", help="File to save the optimized calibration parameters.")
    parser.add_argument("--optim_dsd", action="store_true", help="Flag to optimize DSD parameter.")
    parser.add_argument("--optim_beads", action="store_true", help="Flag to optimize 3D bead coordinates.")
    parser.add_argument("--scheduler",required=False, choices=["steplr", "exponential", "none"], default="none", help="Type of learning rate scheduler to use.")
    args = parser.parse_args()

    calibrate(args.input, args.DSD, args.iter, args.output, False, False, args.lr, args.scheduler)
