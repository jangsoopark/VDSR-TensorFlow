
function image_aug = image_augmentation(image, r, f)

image_aug = imrotate(image, r);

if f == 1 || f == 2
    image_aug = flip(image_aug, f);
end

end

