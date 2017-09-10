#!/bin/bash
 
IMAGES="/home/arun/image-classification/Images"
UNLABELED="./UnlabeledImages.txt"
TRAININGDATA="/home/arun/image-classification/TrainingData"
rm "$UNLABELED"
 
for i in $(ls -1 "$IMAGES")
do
  for j in $(ls -1 "$IMAGES"/"$i")
    do
      rating=`identify -verbose "$IMAGES"/"$i"/"$j" | grep xmp:Rating | cut -d':' -f3`
      rtng=`echo "$rating" | awk '{$1=$1};1'`
      case "$rtng" in
          1) echo " this is cat 1"
             cp "$IMAGES"/"$i"/"$j" "$TRAININGDATA"/cat1/
             ;;
          2) echo "this is cat 2"
             cp "$IMAGES"/"$i"/"$j" "$TRAININGDATA"/cat2/
             ;;
          3) echo "this is cat 3"
             cp "$IMAGES"/"$i"/"$j" "$TRAININGDATA"/cat3/
             ;;
          4) echo "this is cat 4"
             cp "$IMAGES"/"$i"/"$j" "$TRAININGDATA"/cat4/
             ;;
          5) echo "thi is cat 5"
             cp "$IMAGES"/"$i"/"$j" "$TRAININGDATA"/cat5/
             ;;
          *) echo "this is someting else"
             echo "$j" >> "$UNLABELED"
             ;;
      esac
    done
done
