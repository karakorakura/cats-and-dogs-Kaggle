%cats and dogs Kaggle
projectFolder ='E:\CVProject\Sign-Language-to-Speech-master\Sign-Language-to-Speech-master\Neural-network-test-version1\catsANDdogs (copy)';
dogsFolder = [projectFolder , '\data\train\dogs\'];
catsFolder = [projectFolder , '\data\train\cats\'];
dogsFolderV = [projectFolder , '\data\validation\dogs\'];
catsFolderV = [projectFolder , '\data\validation\cats\'];
%%
TrainData=zeros(64,64,3,4000); % 3D with singleton or 4D
imLabel = ones(4000,1);
% Get list of all BMP files in this directory
% DIR returns as a structure array.  You will need to use () and . to get
% the file names.
dogfiles = dir([dogsFolder,'*.jpg']);
nfiles = length(dogfiles)    % Number of files found
cnt=1;
for ii=1:1000
   currentfilename = dogfiles(ii).name;
   currentimage = imread([dogsFolder,currentfilename]);
   TrainData(:,:,:,cnt)=imresize(currentimage ,[64,64]);
   cnt=cnt+1;
end

TrainData(:,:,:,1001:2000)= flip(TrainData(:,:,:,1:1000),2);

cnt = 2001;
catfiles = dir([catsFolder,'*.jpg']);
nfiles = length(catfiles)    % Number of files found
for ii=1:1000
   currentfilename = catfiles(ii).name;
   currentimage = imread([catsFolder,currentfilename]);
   TrainData(:,:,:,cnt)=imresize(currentimage ,[64,64]);
   cnt=cnt+1;
end
%%
TrainData(:,:,:,3001:4000)= flip(TrainData(:,:,:,2001:3000),2);
imLabel(1:2000) = ones(2000,1);
imLabel(2001:4000) = ones(2000,1).*2;
TrainLabels = categorical(imLabel);
%%
%% CNN Architecture
layers = [imageInputLayer([64 64 3]);...
          convolution2dLayer(5,64,'Stride',1,'Padding',2);...
          reluLayer();...
          maxPooling2dLayer(2,'Stride',2);...
          convolution2dLayer(5,128,'Stride',1,'Padding',2);...
          reluLayer();...
          maxPooling2dLayer(2,'Stride',2);...
          convolution2dLayer(5,128,'Stride',1,'Padding',2);...
          reluLayer();...
          convolution2dLayer(5,128,'Stride',1,'Padding',2);...
          reluLayer();...
          convolution2dLayer(5,256,'Stride',1,'Padding',2);...
          reluLayer();...
          maxPooling2dLayer(2,'Stride',2);...
          fullyConnectedLayer(512);...
          reluLayer();...
          dropoutLayer(0.5);...
          fullyConnectedLayer(512);...
          reluLayer();...
          dropoutLayer(0.5);...
          fullyConnectedLayer(2);...
          softmaxLayer();...
          classificationLayer();...
           ];

%%
opts = trainingOptions('sgdm',...
     'LearnRateSchedule','piecewise',...
     'LearnRateDropFactor',0.2,...
     'LearnRateDropPeriod',20,...
     'MaxEpochs',40,...
     'MiniBatchSize',50,...
     'CheckpointPath','E:\TEMP\checkpoint',...
     'ExecutionEnvironment','gpu'...
     );

%%
[trainedNet,traininfo] = trainNetwork(TrainData,TrainLabels,layers,opts);
%%

TestData=zeros(64,64,3,1600); % 3D with singleton or 4D
imLabelTest = ones(1600,1);
% Get list of all BMP files in this directory
% DIR returns as a structure array.  You will need to use () and . to get
% the file names.
dogfilesV = dir([dogsFolderV,'*.jpg']);
nfiles = length(dogfilesV)    % Number of files found
cnt=1;
for ii=1:400
   currentfilename = dogfilesV(ii).name;
   currentimage = imread([dogsFolderV,currentfilename]);
   TestData(:,:,:,cnt)=imresize(currentimage ,[64,64]);
   cnt=cnt+1;
end

TestData(:,:,:,401:800)= flip(TestData(:,:,:,1:400),2);

cnt = 801;
catfilesV = dir([catsFolderV,'*.jpg']);
nfiles = length(catfilesV)    % Number of files found
for ii=1:400
   currentfilename = catfilesV(ii).name;
   currentimage = imread([catsFolderV,currentfilename]);
   TestData(:,:,:,cnt)=imresize(currentimage ,[64,64]);
   cnt=cnt+1;
end

TestData(:,:,:,1201:1600)= flip(TrainData(:,:,:,801:1200),2);
imLabel(1:800) = ones(800,1);
imLabel(801:1600) = ones(800,1).*2;
TestLabels = categorical(imLabelTest);
%%

%% Get Accuracy
acc = 0;

[Ypred,scores] = classify(trainedNet,TestData);

acc = 0;
for i = 1:(1600)
    if Ypred(i) == TestLabels(i)
        acc = acc + 1;
    end
end

(acc/((1600)))*100
%%

