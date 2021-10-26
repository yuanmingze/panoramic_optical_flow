# 1) clean latex 

# rm -rf paper_draft_arXiv

# need python package https://github.com/google-research/arxiv-latex-cleaner
python3 -m arxiv_latex_cleaner_arXiv paper_draft --keep_bib

## 2) resize images
cd paper_draft_arXiv

# resize demo image
# need package imagemagick
imageWidthPixels=300
domoImageFolderList=("./images/of_error/" "./images/of_warp/")
for folder_name in "${domoImageFolderList[@]}"; do
        echo "Resize images in $folder_name"
        for imageFilePath in $(find $folder_name -name '*.jpg'); do
                # file $imageFilePath
                echo $imageFilePath
                convert $imageFilePath -resize $imageWidthPixels $imageFilePath
        done
done

# resize wraparound demo images
imageWidthPixels=450
for imageFilePath in $(find ./images/wraparound/ -name '*.png'); do
        # file $imageFilePath
        echo $imageFilePath
        convert $imageFilePath -resize $imageWidthPixels $imageFilePath
done

# 3) Remove image mate file: *.blend *.xcf *.svg
for imageFilePath in $(find ./images/ -name '*.blend' -or -name '*.xcf' -or -name '*.svg'); do
        # file $imageFilePath
        echo $imageFilePath
        rm $imageFilePath
done
