# Calibration  

- 최초에 output/calibration_results.npy 파일이 없거나, 새로 캘리브레이션이 필요한 경우에만 수행  

폴더 구조:
 ```
    ./data
    ├── calibration
    │   └── 6679
    │         └──botleft_00.png
    │         └── ...
    │         └──botleft_08.png
    │         └──botleft(6679).tif
    │         └──botleft(6679).txt
    │         └──6679_2d.npy  (OUTPUT)
    │         └──6679_3d.npy  (OUTPUT)
    │   └── 6680
    │         └──botright_00.png
    │         └── ...
    │         └──botright_08.png
    │         └──botleft(6680).tif
    │         └──botleft(6680).txt
    │         └──6680_2d.npy  (OUTPUT)
    │         └──6680_3d.npy  (OUTPUT)
    .
    .
    .
 ```

코드 실행:
- make_2d_npy.py  
   - 이미지에 촬영된 비드 위치(x,y) 좌표를 XXXX_2d.npy 파일로 만듬
   - txt_path, npy_save_path 수정 필요
   - (e.g, topleft(6681).txt 참고)

- make_3d_npy.py
   - 파워포인트로 제공받은 3D 위치에서 X,Y 를 X,Z 좌표로 사용하고, 2차원의 y 좌표를 Y 좌표로 사용해 XXXX_3d.npy 파일로 만듬
   - latest_beads_xy, v_values_latest, output_path_latest 수정 필요
   - (e.g, spiralbeads 촬영 geometry-2.pptx 참고)


- calibration.py
   - Calibration 을 위한 한 세트마다(현재 4 세트) 소스를 수정해 npy 파일을 생성
   - (e.g, (6679, 6680, 6681, 6682) 4번 진행하여 4세트의 npy 파일 생성 필요)
```
$ python data\calibration\make_2d_npy.py
$ python data\calibration\make_3d_npy.py
```

```
$ python calibration.py
--> 최종 calibration 결과물은 output/calibration_results.npy 로 저장됨
(e.g, Loss: 67.01536560058594)
```


calibration.py 파라미터: 
```
--DSD (디폴트 값 1100)
--optim_dsd = False
--optim_beads = False
--scheduler
```
만약 --optim_beads를 한다면 optimize된 3d beads 파일도 저장됨

<br><br>

## [240902 데이터]  
https://drive.google.com/drive/folders/1dfA5emcWmJ3lqxDM7yLRvWFF0AL54LfD  
https://docs.google.com/presentation/d/1eHhKQvmNYAsJp0eLN6RImlfypTjJJO4D/edit#slide=id.p4  
