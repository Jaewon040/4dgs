import os
import torch
import numpy as np
from PIL import Image
import argparse
from argparse import Namespace
from tqdm import tqdm
import torch.nn.functional as F
from scene.gaussian_model import GaussianModel, BasicPointCloud
from utils.general_utils import safe_state
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
from gaussian_renderer import render
from utils.render_utils import get_state_at_time
from utils.loss_utils import l1_loss, ssim
from diffusers import StableDiffusionInstructPix2PixPipeline

def edit_4dgs():
    # 명령줄 인자 파싱
    parser = argparse.ArgumentParser(description="4DGS 편집 스크립트")
    
    # 먼저 기본 모델 파라미터 추가
    model_params = ModelParams(parser)
    pipeline_params = PipelineParams(parser)
    opt_params = OptimizationParams(parser)
    hyper_params = ModelHiddenParams(parser)
    
    # 그 다음 편집 관련 파라미터 추가 (기존 인자와 충돌 없이)
    parser.add_argument("--edit_output_path", type=str, required=True, help="편집된 모델을 저장할 경로")
    parser.add_argument("--prompt", type=str, required=True, help="IP2P를 위한 편집 지시문")
    parser.add_argument("--iterations_2", type=int, default=3000, help="최적화 반복 횟수")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="최적화를 위한 학습률")
    parser.add_argument("--num_views", type=int, default=10, help="렌더링할 뷰 수")
    parser.add_argument("--ip2p_guidance", type=float, default=8.5, help="IP2P 가이던스 스케일")
    parser.add_argument("--image_guidance", type=float, default=1.5, help="이미지 가이던스 스케일")
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    os.makedirs(args.edit_output_path, exist_ok=True)
    os.makedirs(os.path.join(args.edit_output_path, "renders_original"), exist_ok=True)
    os.makedirs(os.path.join(args.edit_output_path, "renders_edited"), exist_ok=True)
    os.makedirs(os.path.join(args.edit_output_path, "point_cloud"), exist_ok=True)
    
    # 설정 파일이 제공된 경우 로드
    if hasattr(args, 'configs') and args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    
    # 시스템 상태 초기화
    safe_state(False)
    
    # 모델 로드
    print(f"{args.model_path}에서 모델 로드 중...")
    
    # 체크포인트 로드
    iteration_folders = [f for f in os.listdir(os.path.join(args.model_path, "point_cloud")) if f.startswith("iteration_")]
    if not iteration_folders:
        print("모델 폴더에서 iteration 폴더를 찾을 수 없습니다.")
        return
    
    # 마지막 반복 폴더 선택
    iteration_folders.sort(key=lambda x: int(x.split("_")[1]))
    latest_folder = iteration_folders[-1]
    checkpoint_path = os.path.join(args.model_path, "point_cloud", latest_folder)
    
    # 점 구름 로드
    ply_path = os.path.join(checkpoint_path, "point_cloud.ply")
    
    # jumpingjacks 모델용 하드코딩된 설정
    jumpingjacks_config = {
        'kplanes_config': {
            'grid_dimensions': 2,
            'input_coordinate_dim': 4,
            'output_coordinate_dim': 32,
            'resolution': [64, 64, 64, 100]  # 점프잭 해상도
        },
        'multires': [1, 2],
        'defor_depth': 0,
        'net_width': 64,
        'plane_tv_weight': 0.0001,
        'time_smoothness_weight': 0.01,
        'l1_time_planes': 0.0001,
        'weight_constraint_init': 1,
        'weight_constraint_after': 0.2,
        'weight_decay_iteration': 5000,
        'bounds': 1.6,
        'no_dx': False,
        'no_grid': False,
        'no_ds': False,
        'no_dr': False,
        'no_do': True,
        'no_dshs': True,
        'empty_voxel': False,
        'grid_pe': 0,
        'static_mlp': False,
        'apply_rotation': False,
        'timebase_pe': 4,
        'posebase_pe': 10,
        'scale_rotation_pe': 2,
        'opacity_pe': 2,
        'timenet_width': 64,
        'timenet_output': 32
    }
    
    # 딕셔너리를 Namespace 객체로 변환
    hyper_args = Namespace(**jumpingjacks_config)
    
    # 가우시안 모델 생성
    gaussians = GaussianModel(3, hyper_args)  # 기본 SH 차수 3 사용
    
    # 점 구름 로드
    gaussians.load_ply(ply_path)
    
    # 변형 가중치 로드
    deformation_path = os.path.join(checkpoint_path, "deformation.pth")
    if os.path.exists(deformation_path):
        weight_dict = torch.load(deformation_path, map_location="cuda")
        
        try:
            gaussians._deformation.load_state_dict(weight_dict, strict=False)  # strict=False로 설정해 누락된 키 무시
            print("변형 가중치 로드 성공!")
            
            gaussians._deformation = gaussians._deformation.to("cuda")
            gaussians._deformation_table = torch.gt(torch.ones((gaussians.get_xyz.shape[0]), device="cuda"), 0)
            
            if os.path.exists(os.path.join(checkpoint_path, "deformation_table.pth")):
                gaussians._deformation_table = torch.load(os.path.join(checkpoint_path, "deformation_table.pth"), map_location="cuda")
            
            if os.path.exists(os.path.join(checkpoint_path, "deformation_accum.pth")):
                gaussians._deformation_accum = torch.load(os.path.join(checkpoint_path, "deformation_accum.pth"), map_location="cuda")
            else:
                gaussians._deformation_accum = torch.zeros((gaussians.get_xyz.shape[0], 3), device="cuda")
                
        except RuntimeError as e:
            print(f"변형 가중치 로드 오류: {e}")
            print("모델이 로드될 수 없습니다. 모델 구성을 확인하세요.")
            return
    
    # 렌더링 설정
    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    pipe = pipeline_params.extract(args)
    
    # 카메라 설정 (간단한 테스트 카메라, t=0 고정)
    from scene.cameras import Camera
    
    # 간단한 카메라 설정 (실제로는 원본 데이터셋에서 카메라를 로드해야 함)
    # 이 부분은 실제 데이터셋에 따라 수정 필요
    test_cameras = []
    
    # 임의의 카메라 위치 및 방향 설정
    # 원형으로 배치된 카메라 생성
    fov = 40  # 시야각 (도)
    fov_rad = fov * np.pi / 180  # 라디안으로 변환
    distance = 3.0  # 중심으로부터의 거리
    
    for i in range(args.num_views):
        angle = 2 * np.pi * i / args.num_views
        x = distance * np.sin(angle)
        z = distance * np.cos(angle)
        y = 0.5  # 약간 위에서 내려다 보는 위치
        
        # 카메라는 항상 원점을 바라봄
        look_at = np.array([0, 0, 0])
        position = np.array([x, y, z])
        up = np.array([0, 1, 0])
        
        # 방향 계산
        forward = look_at - position
        forward = forward / np.linalg.norm(forward)
        
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        
        up = np.cross(right, forward)
        
        # 회전 행렬 생성
        R = np.stack([right, up, -forward], axis=0)  # 열 벡터를 쌓아서 행렬 생성
        T = -position @ R  # 카메라 위치를 이동 벡터로 변환
        
        # 카메라 파라미터 설정
        width, height = 800, 800
        fx = fy = height / (2 * np.tan(fov_rad / 2))
        
        camera = Camera(
            colmap_id=i,
            R=R,
            T=T,
            FoVx=fov_rad,
            FoVy=fov_rad,
            image=torch.zeros((3, height, width), device="cuda"),  # 더미 이미지
            gt_alpha_mask=None,
            image_name=f"view_{i:03d}",
            uid=i,
            time=0.0  # 시간은 항상 0으로 설정
        )
        
        test_cameras.append(camera)
    
    # 원본 이미지 렌더링
    original_images = []
    print("원본 이미지 렌더링 중...")
    for i, cam in enumerate(tqdm(test_cameras)):
        # 오류가 발생할 수 있으므로 예외 처리 추가
        try:
            render_pkg = render(cam, gaussians, pipe, background, stage="fine")
            image = render_pkg["render"]
            # 저장을 위해 PIL로 변환
            pil_img = Image.fromarray((image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
            pil_img.save(os.path.join(args.edit_output_path, f"renders_original/view_{i:03d}.png"))
            original_images.append(image)
        except Exception as e:
            print(f"이미지 {i} 렌더링 중 오류 발생: {e}")
            return
    
    # InstructPix2Pix 파이프라인 로드
    print("IP2P 모델 로드 중...")
    ip2p = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        "timbrooks/instruct-pix2pix", torch_dtype=torch.float16
    ).to("cuda")
    
    # IP2P로 이미지 처리
    edited_images = []
    print(f"IP2P로 이미지 편집 중... 프롬프트: '{args.prompt}'")
    for i, img in enumerate(tqdm(original_images)):
        # IP2P를 위해 PIL로 변환
        pil_img = Image.fromarray((img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
        
        # IP2P로 편집
        edited_img = ip2p(args.prompt, image=pil_img, 
                          guidance_scale=args.ip2p_guidance, 
                          image_guidance_scale=args.image_guidance).images[0]
        
        # 편집된 이미지 저장
        edited_img.save(os.path.join(args.edit_output_path, f"renders_edited/view_{i:03d}.png"))
        
        # 텐서로 변환
        edited_tensor = torch.from_numpy(np.array(edited_img).astype(np.float32) / 255.0).permute(2, 0, 1).to("cuda")
        edited_images.append(edited_tensor)
    
    # 최적화 준비 - 포인트 클라우드만 업데이트하고 변형 필드는 업데이트하지 않음
    print("최적화 설정 중...")
    l = [
        {'params': [gaussians._xyz], 'lr': args.learning_rate, "name": "xyz"},
        {'params': [gaussians._features_dc], 'lr': args.learning_rate, "name": "f_dc"},
        {'params': [gaussians._features_rest], 'lr': args.learning_rate / 20.0, "name": "f_rest"},
        {'params': [gaussians._opacity], 'lr': args.learning_rate * 5, "name": "opacity"},
        {'params': [gaussians._scaling], 'lr': args.learning_rate * 0.5, "name": "scaling"},
        {'params': [gaussians._rotation], 'lr': args.learning_rate * 0.1, "name": "rotation"}
    ]
    
    optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
    
    # 저장 및 로딩 유틸리티 함수
    def save_checkpoint(iteration):
        # 모델 저장
        model_dir = os.path.join(args.edit_output_path, "point_cloud", f"iteration_{iteration}")
        os.makedirs(model_dir, exist_ok=True)
        
        # 포인트 클라우드 저장
        gaussians.save_ply(os.path.join(model_dir, "point_cloud.ply"))
        
        # 원본 deformation 관련 파일 복사
        import shutil
        src_dir = os.path.join(args.model_path, "point_cloud")
        
        # 마지막 반복 폴더 찾기
        iteration_dirs = [d for d in os.listdir(src_dir) if d.startswith("iteration_")]
        iteration_dirs.sort(key=lambda x: int(x.split("_")[1]))
        latest_dir = iteration_dirs[-1]
        
        src_deform_path = os.path.join(src_dir, latest_dir, "deformation.pth")
        src_deform_table_path = os.path.join(src_dir, latest_dir, "deformation_table.pth")
        src_deform_accum_path = os.path.join(src_dir, latest_dir, "deformation_accum.pth")
        
        if os.path.exists(src_deform_path):
            shutil.copy(src_deform_path, os.path.join(model_dir, "deformation.pth"))
        if os.path.exists(src_deform_table_path):
            shutil.copy(src_deform_table_path, os.path.join(model_dir, "deformation_table.pth"))
        if os.path.exists(src_deform_accum_path):
            shutil.copy(src_deform_accum_path, os.path.join(model_dir, "deformation_accum.pth"))
    
    # 최적화 루프
    print(f"{args.iterations_2}번의 반복으로 최적화 시작...")
    best_loss = float('inf')
    
    for iteration in tqdm(range(args.iterations_2)):
        # 이번 반복을 위한 카메라 랜덤 선택
        cam_idx = torch.randint(0, len(test_cameras), (1,)).item()
        viewpoint_cam = test_cameras[cam_idx]
        target_image = edited_images[cam_idx]
        
        # 현재 포인트 클라우드 렌더링
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, stage="fine")
        rendered_image = render_pkg["render"]
        
        # 손실 계산 - L1 손실 및 SSIM
        l1 = l1_loss(rendered_image, target_image)
        ssim_loss = 1.0 - ssim(rendered_image.unsqueeze(0), target_image.unsqueeze(0))
        loss = l1 + 0.2 * ssim_loss
        
        # 역전파 및 옵티마이저 스텝
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 중간 저장
        if (iteration + 1) % 500 == 0:
            print(f"반복 {iteration+1}, 손실: {loss.item():.6f}")
            save_checkpoint(iteration + 1)
            
        # 최적의 모델 저장
        current_loss = loss.item()
        if current_loss < best_loss and (iteration + 1) % 100 == 0:
            best_loss = current_loss
            save_checkpoint(iteration + 1)
    
    # 최종 모델 저장
    save_checkpoint(args.iterations_2)
    print(f"최적화 완료. 최종 모델이 {args.edit_output_path}에 저장되었습니다.")
    
    # 최종 결과 렌더링 및 평가
    print("최종 결과 렌더링 중...")
    final_renders_path = os.path.join(args.edit_output_path, "final_renders")
    os.makedirs(final_renders_path, exist_ok=True)
    
    for i, cam in enumerate(tqdm(test_cameras)):
        render_pkg = render(cam, gaussians, pipe, background, stage="fine")
        final_image = render_pkg["render"]
        
        # 원본, 목표, 최종 결과 이미지를 나란히 저장
        original = original_images[i].permute(1, 2, 0).cpu().numpy()
        target = edited_images[i].permute(1, 2, 0).cpu().numpy()
        result = final_image.permute(1, 2, 0).cpu().numpy()
        
        # 이미지 결합
        comparison = np.concatenate([original, target, result], axis=1)
        comparison_img = Image.fromarray((comparison * 255).astype(np.uint8))
        comparison_img.save(os.path.join(final_renders_path, f"comparison_{i:03d}.png"))

if __name__ == "__main__":
    edit_4dgs()