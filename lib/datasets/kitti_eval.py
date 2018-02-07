# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------

import xml.etree.ElementTree as ET
import os,uuid,cv2
import cPickle
import numpy as np

"""
def parse_rec(filename):
    ""Parse a PASCAL VOC xml file ""
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text.strip()
        polygon = obj.find('polygon')

        xs = []
        ys = []

        for pt in polygon.findall('pt'):
            xs += [int(pt.find('x').text)]
            ys += [int(pt.find('y').text)]

        obj_struct['bbox'] = [min(xs),
                               min(ys),
                               max(xs),
                               max(ys)]
        objects.append(obj_struct)

    return objects

"""

def vis_detections(im, class_name, dets, thresh=0.5,name=""):
    """Draw detected bounding boxes."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    if name == "":
        plt.savefig('vis_gt_{}.png'.format(str(uuid.uuid4())))
    else:
        plt.savefig(name.format(str(uuid.uuid4())))        

def parse_rec(filename):
        """
        Load image and bounding boxes info from TXT file in the KITTI
        format.
        """
        with open(filename,"r") as f:
            annos = f.readlines()
        cleaned = [
            obj for obj in annos \
            if "Misc" not in obj.split()[0].strip() and
            "DontCare" not in obj.split()[0].strip()
        ]
        only_people = [
                obj for obj in cleaned \
        if obj.split()[0].strip().lower() in ["pedestrian","person_sitting","cyclist"]]
        annos = only_people

        num_objs = len(annos)
        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        objects = []
        # Load object bounding boxes into a data frame.
        for obj in annos:
            obj = obj.strip().split()
            obj[0] = obj[0].lower()
            if obj[0] in ["pedestrian","person_sitting","cyclist"]:
                obj[0] = "person"
            obj_struct = {}
            obj_struct['name'] = obj[0]
            x1 = float(obj[4])
            y1 = float(obj[5])
            x2 = float(obj[6])
            y2 = float(obj[7])
            obj_struct['bbox'] = [x1, y1, x2, y2]
            objects.append(obj_struct)
        return objects


def voc_ap(rec, prec, clsnm,use_07_metric=False,viz=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if viz:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

    use_07_metric = False

    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))
        if viz:
            plt.plot(mrec,mpre,"g.")

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])


        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

        if viz:
            print(mrec.shape,mpre.shape,ap.shape)
            # plt.plot(mrec,mpre,"ro",mrec,ap,"g^")
            plt.plot(mrec,mpre,"r+")
            plt.tight_layout()
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title("AP: {}".format(ap))
            plt.savefig(clsnm + "_apPlt.png")        
            plt.clf()
            # time.sleep(30)

    return ap

def kitti_eval(detpath,
               annopath,
               imagesetfile,
               classname,
               cachedir,
               ovthresh=0.5,
               use_07_metric=False,
               imagepath=None):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])

    Top level function that does the PASCAL VOC evaluation.

    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(annopath.format(imagename))
            if i % 100 == 0:
                print 'Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(imagenames))
        # save
        print 'Saving cached annotations to {:s}'.format(cachefile)
        with open(cachefile, 'w') as f:
            cPickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'r') as f:
            recs = cPickle.load(f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    n = 0
    vis_gt_ex = False
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        npos = npos + len(bbox)
        det = [False] * len(R)
        class_recs[imagename] = {'bbox': bbox,
                                 'det': det}

        if n < 10 and len(bbox) > 0 and vis_gt_ex == True:
            im = cv2.imread(imagepath.format(imagename))
            vis_detections(im, classname, class_recs[imagename]['bbox'], thresh=0.1)
            n += 1
    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]



    vis_det_ex = True
    if vis_det_ex:
        image_ids_to_idx = {}
        for idx,img_id in enumerate(image_ids):
            if img_id in image_ids_to_idx.keys():
                image_ids_to_idx[img_id] += [idx]
            else:
                image_ids_to_idx[img_id] = [idx]

        n = 10
        ids = np.random.permutation(len(BB))[:n]
        for i,idx in enumerate(ids): # we want "n" unique images
            # we need to grab "class_recs[image_ids[d]]"
            print(i,idx)
            image_id = image_ids[idx]
            # now find all the indicies with the given image_id
            image_idx = image_ids_to_idx[image_id]
            bbox = BB[image_idx,:]
            conf = -1*sorted_scores[image_idx]
            print(conf)
            bboxes = np.concatenate((bbox,conf[:,np.newaxis]),axis=1)
            if len(bboxes) > 0:
                im = cv2.imread(imagepath.format(image_id))
                vis_detections(im, classname, bboxes, thresh=0.20,name="vis_det_{}.png")
                n += 1
            else:
                print("ohno!")
                sys.exit(1)

    ovthresh = [0.5,0.75,0.95]
    nd = len(image_ids)
    tp = np.zeros((nd,len(ovthresh)))
    fp = np.zeros((nd,len(ovthresh)))
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)
        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)
            # print("-=-=-=-=-=-=-=-=-=-=-=-")
            # print(overlaps)
            # print(sorted_scores)
        # if(sorted_scores[d] >= -0.5):
        #     continue
        #print(sorted_scores[d],sorted_scores[d] < -0.0)
        inside_any = False
        for idx in range(len(ovthresh)):
            if ovmax > ovthresh[idx]:
                if not R['det'][jmax]:
                    inside_any = True
                    tp[d,idx] = 1.
                    #print("tp")
                else:
                    fp[d,idx] = 1.
                    #print("fp")
            else:
                fp[d,idx] = 1.
                #print("fp")

        if inside_any is True:
            R['det'][jmax] = 1

    rec = np.zeros((len(fp),len(ovthresh)))
    prec = np.zeros((len(fp),len(ovthresh)))
    ap = np.zeros(len(ovthresh))
    for idx in range(len(ovthresh)):
        # compute precision recall
        _fp = np.cumsum(fp[:,idx])
        _tp = np.cumsum(tp[:,idx])
        rec[:,idx] = _tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec[:,idx] = _tp / np.maximum(_tp + _fp, np.finfo(np.float64).eps)
        #ap = voc_ap(rec, prec, use_07_metric)
        ap[idx] = voc_ap(rec[:,idx], prec[:,idx], classname, False)

    #print(fp,tp,rec,prec,ap,npos)
    return rec, prec, ap, ovthresh


