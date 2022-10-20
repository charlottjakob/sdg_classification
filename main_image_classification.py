from data_preperation.cut_report_pages import seperate_report_pages
from training.creat_image_train_set import create_overlays, add_overlays


# seperate_report_pages(1000)

# create_overlays(0, 2000)

add_overlays(f_from=0, f_to=1000, o_from=0)
add_overlays(f_from=0, f_to=1000, o_from=1000)