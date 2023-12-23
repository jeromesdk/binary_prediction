# mini_project_ML


## Setup project

After copying the repositery in local, use the following command to download required dependancies : 

```
pip install -r requirements.txt
```

## Data descritpion
- data_banknote: 
    - Data format: .txt 
    - Data info: 
        - Dataset Information: Data were extracted from images that were taken from genuine and forged banknote-like specimens.  For digitization, an industrial camera usually used for print inspection was used. The final images have 400x 400 pixels. Due to the object lens and distance to the investigated object gray-scale pictures with a resolution of about 660 dpi were gained. Wavelet Transform tool were used to extract features from images.  
        - Dataset features:    
       1. variance of Wavelet Transformed image (continuous) 
       2. skewness of Wavelet Transformed image (continuous)
       3. curtosis of Wavelet Transformed image (continuous)
       4. entropy of image (continuous)
       5. class (integer) 
       - Number of lines: 1372
       - missing data: No
    - Task: Classification (finding the amount of money of each banknote - class)

- CKD: 
    - Data format: .csv
    - Data info: 
        - Dataset Information : The data was taken over a 2-month period in India, collecting 25 features to classify if patient has a chronik kidney disease or not. It's the dataset we used for GMM if I remember well 
        - Dataset features: 
        1. age (float)
        2. bp  | blood pressure (int)
        3. sg  | (float or category)
        4. al  | (int or category)
        5. su  | (int or category)
        6. rbc | (boolean)
        7. pc  | (boolean)
        8. pcc | (boolean)
        9. ba  | (boolean)
        10. bgr (int)
        11. bu (int (as float in ds))
        12. sc (float)
        13. sod (int (as float in ds))
        14. pot (float)
        15. hemo (float)
        16. pcv (int)
        17. wc (int)
        18. rc (float)
        19. htn (boolean)
        20. dm (boolean)
        21. cad (boolean)
        22. appet (boolean)
        23. pe (boolean)
        24. ane (boolean)
        25. class (boolean)
        - Number of lines: 400 
        - Missing data: yes, a lot
    - Task: Classification (find if patient has ckd or not)




