load('url_data.mat');
p = dir('/home/zzd/DCGAN-tensorflow/market1501_256_48000_large/*.jpg');
%p = dir('/home/zzd/CUHK03/zzd_code/split1_256/*.jpg');
num = numel(imdb.images.data);
for i=1:6000 %numel(p)
    url = strcat('/home/zzd/DCGAN-tensorflow/market1501_256_48000_large/',p(i).name);
    %url = strcat('/home/zzd/CUHK03/zzd_code/split1_256/',p(i).name);
    imdb.images.data(num+i) =cellstr(url);
    imdb.images.label(num+i) = 0;
    imdb.images.set(num+i) = 1;
end

save('url_data_gan_6000.mat','imdb','-v7.3');