for file in ./*
do
	convert "$file" -set colorspace Gray -separate -average "$file"
done
