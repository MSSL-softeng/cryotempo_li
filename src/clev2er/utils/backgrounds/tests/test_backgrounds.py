# """
# pytest tests for cpom.backgrounds
# """

# import logging
# import os

# import matplotlib.pyplot as plt
# import pytest

# from clev2er.utils.areas.area_plot import Polarplot
# from clev2er.utils.areas.areas import Area
# from clev2er.utils.backgrounds.backgrounds import Background, all_backgrounds


# # Setup logging for tests
# log = logging.getLogger(__name__)
# log.setLevel(logging.INFO)


# @pytest.mark.parametrize("background_name", all_backgrounds)
# def test_backgrounds(background_name):
#     """
#     test purpose: for each background, plot background in one or more test areas
#     as specified in  cpom.backgrounds.backgrounds.all_backgrounds which is a dict
#     containing background name as key, and a list of relevant areas to test
#     """

#     # all_backgrounds is a dictionary {'background': [area, area]}
#     test_areas = all_backgrounds[background_name]

#     if not isinstance(test_areas, list):
#         test_areas = [test_areas]

#     plot_dir = f'{os.environ["CPOM_SOFTWARE_DIR"]}/cpom/backgrounds/test/test_images'

#     for area in test_areas:
#         thisarea = Area(area)
#         thisbg = Background(background_name, Area(area))

#         plt.figure(figsize=(10, 10))  # width, height in inches

#         thispolar_plot = Polarplot(thisarea)

#         # Setup geoaxes, extent, projection for area
#         ax, dataprj, circle = thispolar_plot.setup_projection_and_extent()

#         # Load the background
#         thisbg.load(ax, dataprj, alpha=1.0, resolution="low")

#         # Draw coastlines, etc
#         if thisarea.background_color:
#             ax.set_facecolor(thisarea.background_color)
#         thispolar_plot.draw_coastlines(ax, dataprj, None, None)
#         thispolar_plot.draw_gridlines(ax, None, circle)
#         thispolar_plot.draw_mapscale_bar(ax, None, dataprj)
#         thispolar_plot.draw_area_polygon_mask(ax, None, None, dataprj)

#         plt.title(f"Background: {background_name} in area: {area} :")

#         # Save plot
#         plt.savefig(f"{plot_dir}/{background_name}___{area}.png")


# @pytest.mark.parametrize("area", all_areas)
# @pytest.mark.area_test
# def test_all_area_backgrounds(area):
#     """
#     test purpose: for each area, plot the associated background given by Area.background_image
#     """

#     # all_backgrounds is a dictionary {'background': [area, area]}

#     plot_dir = f'{os.environ["CPOM_SOFTWARE_DIR"]}/cpom/backgrounds/test/test_area_images'

#     thisarea = Area(area)
#     if not thisarea.background_image:
#         assert False, "No background image for area"

#     plt.figure(figsize=(10, 10))  # width, height in inches

#     thispolar_plot = Polarplot(thisarea)

#     # Setup geoaxes, extent, projection for area
#     ax, dataprj, circle = thispolar_plot.setup_projection_and_extent()

#     if isinstance(thisarea.background_image, list):
#         backgrounds = thisarea.background_image
#     else:
#         backgrounds = [thisarea.background_image]

#     if isinstance(thisarea.background_image_alpha, list):
#         alphas = thisarea.background_image_alpha
#     else:
#         alphas = [thisarea.background_image_alpha]

#     if isinstance(thisarea.background_image_resolution, list):
#         resolutions = thisarea.background_image_resolution
#     else:
#         resolutions = [thisarea.background_image_resolution]

#     assert len(backgrounds) == len(alphas)
#     assert len(backgrounds) == len(resolutions)

#     for background, alpha, resolution in zip(backgrounds, alphas, resolutions):
#         # Load the background
#         Background(background, Area(area)).load(ax, dataprj, alpha=alpha, resolution=resolution)

#     # Draw coastlines, etc
#     if thisarea.background_color:
#         ax.set_facecolor(thisarea.background_color)
#     thispolar_plot.draw_coastlines(ax, dataprj, None, None)
#     thispolar_plot.draw_gridlines(ax, None, circle)
#     thispolar_plot.draw_mapscale_bar(ax, None, dataprj)
#     thispolar_plot.draw_area_polygon_mask(ax, None, None, dataprj)

#     plt.title(f"Background: {thisarea.background_image} in area: {area} :")

#     # Save plot
#     plt.savefig(f"{plot_dir}/{thisarea.background_image}___{area}.png")
