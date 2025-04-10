import os
import torch
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm
import torch.nn.functional as F
from scene import Scene
from scene.gaussian_model import GaussianModel, BasicPointCloud
from utils.general_utils import safe_state
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
from gaussian_renderer import render
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
    model_args = model_params.extract(args)
    
    # 가우시안 모델 생성 및 로드
    gaussians = GaussianModel(3, hyper_params.extract(args))  # 기본 SH 차수 3 사용
    scene = Scene(model_args, gaussians)
    
    # 렌더링 설정
    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    pipe = pipeline_params.extract(args)
    
    # t=0 카메라 가져오기
    train_cameras = scene.getTrainCameras()
    t0_cameras = []
    for i, cam in enumerate(train_cameras):
        if i < args.num_views:  # 지정된 수의 카메라만 가져오기
            cam.time = 0.0  # 시간을 0으로 설정
            t0_cameras.append(cam)
    
    # 원본 t=0 이미지 렌더링
    original_images = []
    print("원본 이미지 렌더링 중...")
    for i, cam in enumerate(tqdm(t0_cameras)):
        render_pkg = render(cam, gaussians, pipe, background, stage="fine")
        image = render_pkg["render"]
        # 저장을 위해 PIL로 변환
        pil_img = Image.fromarray((image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
        pil_img.save(os.path.join(args.edit_output_path, f"renders_original/view_{i:03d}.png"))
        original_images.append(image)
    
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
    print(f"{args.iterations}번의 반복으로 최적화 시작...")
    best_loss = float('inf')
    
    for iteration in tqdm(range(args.iterations)):
        # 이번 반복을 위한 카메라 랜덤 선택
        cam_idx = torch.randint(0, len(t0_cameras), (1,)).item()
        viewpoint_cam = t0_cameras[cam_idx]
        target_image = edited_images[cam_idx]
        
        # 현재 포인트 클라우드 렌더링
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, stage="fine")
        rendered_image = render_pkg["render"]
        
        # 손실 계산 - L1 손실 및 SSIM 손실 조합
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
    save_checkpoint(args.iterations)
    print(f"최적화 완료. 최종 모델이 {args.edit_output_path}에 저장되었습니다.")
    
    # 최종 결과 렌더링 및 평가
    print("최종 결과 렌더링 중...")
    final_renders_path = os.path.join(args.edit_output_path, "final_renders")
    os.makedirs(final_renders_path, exist_ok=True)
    
    for i, cam in enumerate(tqdm(t0_cameras)):
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