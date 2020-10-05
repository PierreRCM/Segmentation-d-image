import numpy as np
import matplotlib.image as im
import matplotlib.pyplot as plt
import matplotlib.patches as pat
from scipy import ndimage


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def draw_rectangle(region_seg, ax, fill=False, color="r"):

    for pave in region_seg:
        rect = pat.Rectangle((pave[2], pave[0]), pave[3] - pave[2], pave[1] - pave[0], linewidth=0.4, fill=fill,
                             color=color)
        ax.add_patch(rect)


def which_pave(region_segmented, coord):

    result = None
    for pave in region_segmented:
        if pave[0] < coord[0] < pave[1] and pave[2] < coord[1] < pave[3]:

            result = pave

    return result


def Mahalanobis(new_voisins_to_test, group_fusionner_moments, mahalanobis):

    voisin_valide = []
    # fig, ax = plt.subplots(1)
    for rect_voisin in new_voisins_to_test:

        if (rect_voisin[5] + group_fusionner_moments[1]) < 1.0E-16:
            mahalanobis_distance = (rect_voisin[4] - group_fusionner_moments[0]) ** 2 / 1.0E-16
            # print((rect_voisin[4] - group_fusionner_moments[0]) ** 2)
        else:
            mahalanobis_distance = (rect_voisin[4] - group_fusionner_moments[0])**2 / \
                                   (rect_voisin[5] + group_fusionner_moments[1])

        # draw_rectangle([rect_voisin], ax, fill=True, color="b")
        # print(mahalanobis_distance)
        if mahalanobis_distance < mahalanobis:
            voisin_valide.append(rect_voisin)

    # draw_rectangle(rect_fusionner, ax, fill=True)
    # plt.imshow(mon_image, cmap="gray")
    # plt.show()
    return voisin_valide


def compute_new_moments(voisins_to_add, group_fusionner_moments):

    mean_g = 0
    variance_g = 0
    Ntot = 0

    for rect_to_add in voisins_to_add:

        Ngr = group_fusionner_moments[2]
        Na = rect_to_add[6]
        Ntot = Ngr + Na

        mean_g = (Na * rect_to_add[4] + Ngr * group_fusionner_moments[0]) / Ntot  # new mean

        variance_g = (Na * rect_to_add[4]**2 + Ngr * group_fusionner_moments[0]**2
                                     - Ntot * mean_g**2 + (Na - 1) * rect_to_add[5] + (Ngr - 1) *
                                     group_fusionner_moments[1]) / (Ntot - 1)

    return [mean_g, variance_g, Ntot]


def select_voisins(regions, new_voisins):

    list_rect_voisin = []
    list_indice_voisin = []

    for rect_selected in new_voisins:
        for i, rect_to_test in enumerate(regions):

            if voisins(rect_selected, rect_to_test):
                if list(rect_to_test) not in [list(x) for x in list_rect_voisin]:

                    list_rect_voisin.append(rect_to_test)
                    list_indice_voisin.append(i)
            else:
                pass

    return list_rect_voisin, list_indice_voisin


def remove_from_region(region_s, indices_des_voisins):

    region_s = np.delete(region_s, indices_des_voisins, 0)

    return region_s


def onclick(event):

    global col, lin

    lin = event.xdata
    col = event.ydata

    plt.close()


def voisins(rectangle1, rectangle2):

    voisin = False

    if rectangle1[0] == rectangle2[1]:
        if rectangle1[2] <= (rectangle2[3]+rectangle2[2])/2.0 <= rectangle1[3] or rectangle2[2] <= (rectangle1[3]+rectangle1[2])/2.0 <= rectangle2[3]:
            voisin = True

    elif rectangle1[2] == rectangle2[3]:
        if rectangle1[0] <= (rectangle2[1]+rectangle2[0])/2.0 <= rectangle1[1] or rectangle2[0] <= (rectangle1[1]+rectangle1[0])/2.0 <= rectangle2[1]:
            voisin = True

    elif rectangle1[1] == rectangle2[0] :
        if rectangle1[2] <= (rectangle2[3]+rectangle2[2])/2.0 <= rectangle1[3] or rectangle2[2] <= (rectangle1[3]+rectangle1[2])/2.0 <= rectangle2[3]:
            voisin = True

    elif rectangle1[3] == rectangle2[2]:
        if rectangle1[0] <= (rectangle2[1] + rectangle2[0]) / 2.0 <= rectangle1[1] or rectangle2[0] <= (rectangle1[1] + rectangle1[0]) / 2.0 <= rectangle2[1]:
            voisin = True

    return voisin


def momentsImage(monimage, pave):
    moy = np.mean(monimage[pave[0]:pave[1], pave[2]:pave[3]])
    var = np.var(monimage[pave[0]:pave[1], pave[2]:pave[3]])
    nbpoint = monimage[pave[0]:pave[1], pave[2]:pave[3]].size

    return moy, var, nbpoint


def segmentation(monimage, regions, area=4, seuil=100):

    region_temp = regions.copy()
    global timer
    for i, pave in enumerate(regions):
        moy, var, nbpoint = momentsImage(monimage, pave.copy())
        pave[4] = moy  # For each pave, insert mean, variance, nbpoint
        pave[5] = var
        pave[6] = nbpoint
        area_pave = (pave[1]-pave[0])*(pave[3]-pave[2])
        region_temp[i, :] = pave

        if var > seuil and area_pave > area:
            pave1 = [pave[0], int((pave[0] + pave[1]) / 2), pave[2], int((pave[2] + pave[3]) / 2), 0, 0, 0]
            pave2 = [int((pave[0] + pave[1]) / 2), pave[1], pave[2], int((pave[2] + pave[3]) / 2), 0, 0, 0]
            pave3 = [pave[0], int((pave[0] + pave[1]) / 2), int((pave[2] + pave[3]) / 2), pave[3], 0, 0, 0]
            pave4 = [int((pave[0] + pave[1]) / 2), pave[1], int((pave[2] + pave[3]) / 2), pave[3], 0, 0, 0]
            region_temp[i, :] = pave1
            region_temp = np.concatenate((region_temp, [pave2, pave3, pave4]))

    # fig, ax = plt.subplots(1)  # create figure
    # draw_rectangle(region_temp, ax)  # Draw rect on figure
    # plt.imshow(mon_image, cmap="gray")  # plot image
    # plt.show()
    # timer += time.time() - k
    if region_temp.shape[0] != regions.shape[0]:

        result = segmentation(monimage, region_temp, area, seuil)
    else:
        result = region_temp.copy()

    return result


timer = 0
col = 0
lin = 0
list_timer = []
ima = im.imread("bacterie.jpg")
mon_image =  rgb2gray(ima)
sobelx = ndimage.sobel(mon_image, axis=0, mode="constant")
sobely = ndimage.sobel(mon_image, axis=1, mode="constant")
mon_image = np.hypot(sobelx, sobely)
a, b  = np.histogram(mon_image)
# plt.plot(a)
# plt.show()

mon_image *= 255.0 / np.max(mon_image)

mon_image = mon_image.astype(int)
for i in range(mon_image.shape[0]):
    for j in range(mon_image.shape[1]):
        if mon_image[i][j] > 55:
            mon_image[i][j] = 255
        else :
            mon_image[i][j] = 0
plt.imshow(mon_image, cmap="gray")
plt.show()
Nline, Ncol = mon_image.shape

moyenne, variance, points = momentsImage(mon_image, [0, Nline, 0, Ncol, ])
region = np.array([[0, Nline, 0, Ncol, moyenne, variance, points]], dtype=int)

region_seg = segmentation(mon_image, region)  # segment the image

fig, ax = plt.subplots(1)  # create figure
draw_rectangle(region_seg, ax)  # Draw rect on figure
a = fig.canvas.mpl_connect("button_press_event", onclick)  # enable clicking
plt.imshow(mon_image, cmap="gray")  # plot image

copie_region_seg = region_seg.copy()
plt.show()

pave_select = which_pave(region_seg, [col, lin])
Mg, Vg, Ng = pave_select[4], pave_select[5], pave_select[6]  # mean, variance, nbpoint of group
moment_du_groupe_fusionner = [Mg, Vg, Ng]
rect_fusionner = np.array([pave_select])
last_voisin = rect_fusionner.copy()

voisin_to_merge_with_group = [1]


list_mahalanobis = [1.5, 2, 3, 0.5]


while len(voisin_to_merge_with_group) > 0:

    voisin_select, indices_voisins = select_voisins(region_seg.copy(), last_voisin)

    region_seg = remove_from_region(region_seg.copy(), indices_voisins)

    voisin_to_merge_with_group = Mahalanobis(voisin_select, moment_du_groupe_fusionner, mahalanobis=0.8)

    moment_du_groupe_fusionner = compute_new_moments(voisin_to_merge_with_group, moment_du_groupe_fusionner)

    if len(voisin_to_merge_with_group) != 0:
        rect_fusionner = np.concatenate((rect_fusionner, np.array(voisin_to_merge_with_group)))

    last_voisin = voisin_to_merge_with_group

fig, ax = plt.subplots(1)
draw_rectangle(region_seg, ax, fill=True)
plt.imshow(ima)
plt.show()
region_seg = copie_region_seg.copy()




