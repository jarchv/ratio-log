clc;
clear all;

img_inp_dir = dir('/home/jose/Documents/foo/venvs/pytorch/code/reps/model-exp/datasets/FAIDres/test/*/*flash.png');
root_result = '../../exp/SRIE/';

if ~exist(root_result, 'dir')
    mkdir(root_result);
end

tsum = 0.0;
for k=1:numel(img_inp_dir)
    t = tic;
    X = demoFn(strcat(img_inp_dir(k).folder,"/",img_inp_dir(k).name));
    imwrite(X, strcat(root_result, strrep(img_inp_dir(k).name, "flash.png", "synth.png")))
    d = toc(t);
    tsum = tsum + d;
    fprintf("%f sec\n", d);
end
fprintf("%f sec in avg\n", tsum / 300.0);
