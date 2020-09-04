clc;clear;

img_inp_dir = dir('/home/jose/Documents/foo/venvs/pytorch/code/reps/model-exp/datasets/FAIDres/valid/*/*ambient.png');
%root_result = '../../exp/GC-2.0/eval/';
root_result = '../../exp/Opt/eval/'

if ~exist(root_result, 'dir')
    mkdir(root_result);
end

tsum = 0.0;
gamma = 2.0;

for k=1:numel(img_inp_dir)
    t = tic;
    I = imread(strcat(img_inp_dir(k).folder,"/",img_inp_dir(k).name));
    %X = uint8(255 * (double(I)/255).^(1/gamma));
   
    imwrite(I, strcat(root_result, strrep(img_inp_dir(k).name, "ambient.png", "synth.png")))
    d = toc(t);
    tsum = tsum + d;
    fprintf("%f sec\n", d);
end
fprintf("%f sec in avg\n", tsum / 300.0);