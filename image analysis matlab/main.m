clear all
clc
close all

%%%% parametrized windows
parametrizedSize = [50 100 200 400 600 800;50 100 341 455 683 1366];

for paramI2 = 1 : size(parametrizedSize,2)
    for paramI = 1 : size(parametrizedSize,2)
        textFileName = sprintf('valuesForSize%dby%d.txt', parametrizedSize(1,paramI2) , parametrizedSize(2,paramI));%create files w.r.t different window sizes
        
        secondTextFileName = sprintf('valuesForSize%dby%d_histogram.txt', parametrizedSize(1,paramI2) , parametrizedSize(2,paramI));
        
        fid = fopen(textFileName, 'at'); % Open for writing
        fprintf(fid, 'imageID;');  %name of columns
        fprintf(fid, 'text densities;');
        fprintf(fid, 'Histogram of windows with text densitites b/w 0:0.1:1;');
        fprintf(fid, 'threshold;');
        fprintf(fid, 'windows greater than threshold;');
        fprintf(fid, 'largest connected window (no. of windows);');
        fprintf(fid, 'largest connected window in pixels;');
        fprintf(fid,'The average size of connected regions;');
        fprintf(fid, '\n');
        
        fid2 = fopen(secondTextFileName, 'at'); % Open for writing
        fprintf(fid2 , 'Histogram of connected region;');
        fprintf(fid2 , '\n');
        
        fclose(fid);
        fclose(fid2);
    end
end

myFolderInfo = dir('Screens'); %searching for the folder in which images are put
numberOfFiles = size(myFolderInfo,1);
for fileNumber = 3:numberOfFiles
    I = imread(strcat('Screens\',myFolderInfo(fileNumber).name));
    imageFileName = myFolderInfo(fileNumber).name;  %load the next image file in the folder
    disp('\n\nDOING file number this:');disp(fileNumber);
    disp(imageFileName);
    close all;
    % Perform OCR
    I = rgb2gray(I);

    I = imresize(I , [3588 1366]);  %common size to make sure the number of observation for any image are same.
                                    
    results = ocr(I);
    %figure;
    %imshow(I);
    %locate the text in the image and see how much pixels it covers
    textPart = 0;
    for n = 1:size(results.Words,1)
        wordBBox = results.WordBoundingBoxes(n,:);
        textPart = textPart + (wordBBox(3) * wordBBox(4));
        rectangle('Position',wordBBox,'EdgeColor','g','LineWidth',2)
    end
    disp(textPart)
    size(I)
    ratio = textPart / (size(I,1) * size(I,2));
    disp(ratio)
    disp(size(results.Text))
    %%%%make each letter become a rectangle
    myBWImage2 = zeros(size(I));
    for n = 1: size(results.Words,1)
        A = cell2mat(isstrprop(results.Words(n), 'alphanum'));
        if(sum(sum(A)) ~= 0)
            wordBBox = results.WordBoundingBoxes(n,:);
            for i = wordBBox(2):wordBBox(2) + wordBBox(4)
                for j = wordBBox(1):wordBBox(1) + wordBBox(3)
                    myBWImage2(i,j) = 1;
                end
            end
        end
    end
    %Join the rectangles to make wentences and paragraphs
    myBWImage = im2bw(myBWImage2);
    %figure();
    %imshow(myBWImage);
    se = strel('rectangle',[7,10]);
    closeBW = imclose(myBWImage,se);
    %figure()
    %imshow(closeBW);
    %%%%% white the bounding box of each text box!!!
    CC = bwconncomp(closeBW);
    s  = regionprops(CC,'BoundingBox');

    myBWImage3 = zeros(size(I));
    for n = 1: size(s,1)
        wordBBox = ceil(s(n).BoundingBox);
        for i = wordBBox(2):wordBBox(2) + wordBBox(4)
            for j = wordBBox(1):wordBBox(1) + wordBBox(3)
                myBWImage3(i,j) = 1;
                end
            end
    end
    figure()
    imshow(myBWImage3);

    values = 0;
    for paramI2 = 1 : size(parametrizedSize,2)
        for paramI = 1 : size(parametrizedSize,2)
            fprintf('size %d by %d\n:',parametrizedSize(1,paramI2),parametrizedSize(2,paramI));
            textFileName = sprintf('valuesForSize%dby%d.txt', parametrizedSize(1,paramI2) , parametrizedSize(2,paramI));
            fid = fopen(textFileName, 'at'); % Open for writing
            fprintf(fid, imageFileName);
            fprintf(fid, ';');
            
            secondTextFileName = sprintf('valuesForSize%dby%d_histogram.txt', parametrizedSize(1,paramI2) , parametrizedSize(2,paramI));
            fid2 = fopen(secondTextFileName, 'at'); % Open for writing
            fprintf(fid2, imageFileName);
            fprintf(fid2, ';');
            
            
            windowRows = floor(size(myBWImage3,1) / parametrizedSize(1,paramI2));
            windowCols = floor(size(myBWImage3,2) / parametrizedSize(2,paramI));
            numberOfWinRows = parametrizedSize(1,paramI2);
            numberOfWinCols = parametrizedSize(2,paramI);
            %remainingRows = rem(size(closeBW,1),windowRows);
            %remainingCols = rem(size(closeBW,2),windowCols);
            values2 = zeros(1, ((floor(windowRows)) * (floor(windowCols))));
            for m = 0 : (floor(windowRows) - 1)
                for n = 0 : (floor(windowCols) - 1)
                    whitePixels = 0;
                    for i = 1 : numberOfWinRows
                        for j = 1 : numberOfWinCols
                            if(closeBW(i + (m * numberOfWinRows) , j + (n * numberOfWinCols)) == 1)
                                whitePixels = whitePixels + 1;
                            end
                        end
                    end
                    values2(1, (m + 1)* (n + 1)) = whitePixels;
                end
            end
            
            %values2 = values2(1,2:size(values2,2));
            values3 = zeros(windowRows,windowCols);
            values4 = zeros(windowRows,windowCols);
            for i  =  1 : windowRows
                for j = 1 : windowCols
                    values3(i,j) = values2(1,(i-1)*windowCols + j);
                end
            end

            for i  =  1 : windowRows
                for j = 1 : windowCols
                    if( (values3(i,j) - (numberOfWinRows * numberOfWinCols))~= 0)
                        values4(i,j) = abs(values3(i,j)/ ((numberOfWinRows * numberOfWinCols)));
                    else
                        values4(i,j) = 1;
                    end
                end
            end
            %disp('vlaues4 are ');
            %disp(values4);
% % %             fprintf(fid, 'the parameter size are %d by %d', parametrizedSize(1,paramI2) , parametrizedSize(2,paramI));
% % %             fprintf(fid, '\n');
% % %             fprintf(fid, ';');
            for i=1:size(values4,1)
                fprintf(fid, '%2.4f,', values4(i,:));
            end
            fprintf(fid, ';');
            
            %textDensity histogram
            
            textDensities1D = reshape(values4',1,(size(values4,1) * size(values4,2)));
            binranges = 0.01:0.1:1.01;
            cfdTextDensities  = (histc(textDensities1D,binranges))';
            for i=1:size(cfdTextDensities,1)
                fprintf(fid, '%d,', cfdTextDensities(i));
            end
            fprintf(fid,';');
            
            fprintf('density and CFD density done\n:');
            
            threshold = mean(nonzeros(values4));
            values5 = zeros(size(values4));
            numberOfWinGr8rThreshold = 0;
            for i = 1: size(values4,1)
                for j = 1 : size(values4,2)
                    if(values4(i,j) >= threshold)
                        numberOfWinGr8rThreshold = numberOfWinGr8rThreshold + 1;
                        values5(i,j) = 1;
                    end
                end
            end
            fprintf(fid,'%2.4f;' , threshold);
            fprintf(fid, '%d;', numberOfWinGr8rThreshold);
            
            connComp = bwlabel(logical(values5)); %find the connected components
            imageStats = regionprops(connComp,'BoundingBox'); 
            compNumber = size(imageStats,1);
            if(compNumber ~= 0)
                box1 = imageStats(1).BoundingBox;
                compareVar1 = box1(3)*box1(4);
                largestPosition = 1;
                sizeOfAllConnectedRegion = 0;
                for i=1:compNumber % to compare sizes of connected components
                    box2 = imageStats(i).BoundingBox;
                    compareVar2 = box2(3)*box2(4);
                    bogus = 0;
                    box2 = ceil(box2);
                    %%the CORRECT size of every window
                    for alpha = 0: (box2(4)-1) %%along rows
                        for beta = 0 : (box2(3)-1)%%along columns of the box of connected region
                            if(values5(box2(2) + alpha,box2(1) + beta) == 1)
                                bogus = bogus + 1; 
                            end
                        end
                    end
                    
                    sizeOfAllConnectedRegion = [sizeOfAllConnectedRegion,bogus];
                    if compareVar1 < compareVar2
                        box1 = box2;
                        compareVar1 = compareVar2;
                        largestPosition=i;
                    end
                end
                sizeOfLargestConnectedRegion = 0;
                connectedRegionIndices = ceil(imageStats(largestPosition).BoundingBox);
                %%count windows in the connected region
                for i = 0: (connectedRegionIndices(4)-1) %%along rows
                    for j = 0 : (connectedRegionIndices(3)-1)%%along columns of the box of connected region
                        if(values5(connectedRegionIndices(2) + i,connectedRegionIndices(1) + j) == 1)
                           sizeOfLargestConnectedRegion = sizeOfLargestConnectedRegion + 1; 
                        end
                    end
                end
                fprintf(fid,'%d;',sizeOfLargestConnectedRegion);
                fprintf(fid, '%d;',sizeOfLargestConnectedRegion * parametrizedSize(1,paramI2) * parametrizedSize(2,paramI));
                
                %histogram of windows sizes
                sizeOfAllConnectedRegion = sizeOfAllConnectedRegion(1,2:size(sizeOfAllConnectedRegion,2));
                if(size(sizeOfAllConnectedRegion,2) > 1)
                    a = unique(sizeOfAllConnectedRegion);
                    out = [a',(histc(sizeOfAllConnectedRegion(:),a))];
                else
                    out = [sizeOfAllConnectedRegion,1];
                end
                for counter = 1:size(out,1)
                    %%size of the window by how many windows
                    fprintf(fid2,'%d-%d,',out(counter,2),out(counter,1));
                end
                fprintf(fid2,';');
                 
                out2 = zeros(size(out,1),1);
                for counter = 1:size(out,1)
                    out2(counter,1) = out(counter,1) .* out(counter,2);
                end
                fprintf(fid,'%2.2f;',mean(out2));
                
                fprintf('ALLES good!!!!! %s\n', imageFileName);
            end
            fprintf(fid,'\n');
            fprintf(fid2,'\n');
            fclose(fid);
            fclose(fid2);
        end
    end
end