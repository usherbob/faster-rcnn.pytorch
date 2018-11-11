class AnchorTargetLayer(caffe.Layer):
    """
    Assign anchors to ground-truth targets. Produces anchor classification
    labels and bounding-box regression targets.
    """

    def setup(self, bottom, top):
        # print '~~~~~~~~~~~~self.param_str_: {}'.format(self.param_str_) # 'feat_stride':16
        layer_params = yaml.load(self.param_str_)
        # 字典的get方法，取字典元素，比[]更强大：采用[]，如果所取元素不在字典里面则报错；而get方法不会报错，而是指定了
        # 默认值
        anchor_scales = layer_params.get('scales', (8, 16, 32))
        # 调用generate_anchors方法生成最初始的9个anchor，也就是位于图像最左上角的那个位置的9个；后面会根据shifits来
        # 确定其他的anchors
        self._anchors = generate_anchors(scales=np.array(anchor_scales))
        self._num_anchors = self._anchors.shape[0]
        self._feat_stride = layer_params['feat_stride']

        if DEBUG:
            print 'anchors:'
            print self._anchors
            print 'anchor shapes:'
            print np.hstack((
                self._anchors[:, 2::4] - self._anchors[:, 0::4],
                self._anchors[:, 3::4] - self._anchors[:, 1::4],
            ))
            # 初始化一些数据属性，后面在求bbox的mean和std时会用到
            self._counts = cfg.EPS
            self._sums = np.zeros((1, 4))
            self._squared_sums = np.zeros((1, 4))
            self._fg_sum = 0
            self._bg_sum = 0
            self._count = 0

        # allow boxes to sit over the edge by a small amount
        # 设为0，则取出任何超过图像边界的proposals，只要超出一点点，都要去除
        self._allowed_border = layer_params.get('allowed_border', 0) 
        if DEBUG:
             print '~~~~~~~~~~~~~~~~anchor_scales: {}'.format(anchor_scales) # (8, 16, 32)
             print '~~~~~~~~~~~~~~~~self._feat_stride: {}'.format(self._feat_stride) # 16
             print '~~~~~~~~~~~~~~~~self._allowed_border: {}'.format(self._allowed_border) # 0

        height, width = bottom[0].data.shape[-2:]
        if DEBUG:
            print 'AnchorTargetLayer: height', height, 'width', width # bottom[0] is rpn_cls_score ,so height is 39, width is 64

        A = self._num_anchors
        # labels
        top[0].reshape(1, 1, A * height, width)
        # bbox_targets
        top[1].reshape(1, A * 4, height, width)
        # bbox_inside_weights
        top[2].reshape(1, A * 4, height, width)
        # bbox_outside_weights
        top[3].reshape(1, A * 4, height, width)

    def forward(self, bottom, top):
        # Algorithm:
        #
        # for each (H, W) location i
        #   generate 9 anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the 9 anchors
        # filter out-of-image anchors
        # measure GT overlap

        assert bottom[0].data.shape[0] == 1, \
            'Only single item batches are supported'

        # map of shape (..., H, W)
        height, width = bottom[0].data.shape[-2:]
        # GT boxes (x1, y1, x2, y2, label)
        gt_boxes = bottom[1].data
        # print '~~~~~~~~~~~~~~~~~~~gt_boxes.shape {}'.format(gt_boxes.shape)
        # im_info
        im_info = bottom[2].data[0, :]

        if DEBUG:
            print ''
            print 'im_size: ({}, {})'.format(im_info[0], im_info[1])
            print 'scale: {}'.format(im_info[2])
            print 'height, width: ({}, {})'.format(height, width)
            print 'rpn: gt_boxes.shape', gt_boxes.shape
            print 'rpn: gt_boxes', gt_boxes

        # 1. Generate proposals from bbox deltas and shifted anchors
        # x轴（也就是width）上的偏移值，以600 × 1000的图像为例，会有64个偏移值，因为width=64

        shift_x = np.arange(0, width) * self._feat_stride
        # y轴（也就是height）上的偏移值，以600 × 1000的图像为例，会有39个偏移值，因为width=39
        shift_y = np.arange(0, height) * self._feat_stride

        # shift_x，shift_y均为39×64的二维数组，对应位置的元素组合即构成图像上需要偏移量大小（偏移量大小是相对与图像最
        # 左上角的那9个anchor的偏移量大小），也就是说总共会得到2496个偏移值对。这些偏移值对与初始的anchor相加即可得到
        # 所有的anchors，所以对于600×1000的图像，总共会产生2496×9个anchors，且存储在all_anchors变量中
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        if DEBUG:
             print '~~~~~~~~~~~~~~~~~~shift_x: {}'.format(shift_x)
             print '~~~~~~~~~~~~~~~~~~shift_y: {}'.format(shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                            shift_x.ravel(), shift_y.ravel())).transpose()
        if DEBUG:
             print '~~~~~~~~~~~~~~~~~~shift_x: {}'.format(shift_x)
             print '~~~~~~~~~~~~~~~~~~shift_y: {}'.format(shift_y)
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        A = self._num_anchors
        K = shifts.shape[0]
        if DEBUG:
             print '~~~~~~~~~~~~~~~~~self._anchors.reshape((1, A, 4)).shape:{}'.format(self._anchors.reshape((1, A, 4)).shape)
         print '~~~~~~~~~~~~~~~~~shifts.reshape((1, K, 4)).transpose((1, 0, 2)).shape :{}'.format(shifts.reshape((1, K, 4)).transpose((1, 0, 2)).shape)
        all_anchors = (self._anchors.reshape((1, A, 4)) +
                       shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
        all_anchors = all_anchors.reshape((K * A, 4))
        total_anchors = int(K * A)

        # only keep anchors inside the image 去除超过图像边界的proposals
        inds_inside = np.where(
            (all_anchors[:, 0] >= -self._allowed_border) &
            (all_anchors[:, 1] >= -self._allowed_border) &
            (all_anchors[:, 2] < im_info[1] + self._allowed_border) &  # width
            (all_anchors[:, 3] < im_info[0] + self._allowed_border)    # height
        )[0]

        #total_anchors 18252
        #inds_inside 6136
        #total_anchors 20358
        #inds_inside 7148

        if DEBUG:
            print 'total_anchors', total_anchors
            print 'inds_inside', len(inds_inside)

        # keep only inside anchors
        anchors = all_anchors[inds_inside, :]
        if DEBUG:
            print 'anchors.shape', anchors.shape #(6136,4)  (7148,4)

        # label: 1 is positive, 0 is negative, -1 is dont care
        labels = np.empty((len(inds_inside), ), dtype=np.float32)
        labels.fill(-1)

        # overlaps between the anchors and the gt boxes
        # overlaps (ex, gt) 这里的anchor已经取出了那些超过图像边界的anchors了，使得anchor的数量有～20000 
        # 减少为～6000
        overlaps = bbox_overlaps(
            np.ascontiguousarray(anchors, dtype=np.float),
            np.ascontiguousarray(gt_boxes, dtype=np.float))
        argmax_overlaps = overlaps.argmax(axis=1)
        max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
        gt_argmax_overlaps = overlaps.argmax(axis=0)
        gt_max_overlaps = overlaps[gt_argmax_overlaps,
                                   np.arange(overlaps.shape[1])]
        if DEBUG:
             print '~~~~~~~~~~~~~~~overlaps.shape :{}'.format(overlaps.shape)
             print '~~~~~~~~~~~~~~~overlaps: {}'.format(overlaps)
             print '~~~~~~~~~~~~~~~argmax_overlaps: {}'.format(argmax_overlaps)
             print '~~~~~~~~~~~~~~~max_overlaps.shape: {}'.format(max_overlaps.shape)
             print '~~~~~~~~~~~~~~~max_overlaps: {}'.format(max_overlaps)
             print '~~~~~~~~~~~~~~~gt_argmax_overlaps: {}'.format(gt_argmax_overlaps)
             print '~~~~~~~~~~~~~~~gt_max_overlaps.shape :{}'.format(gt_max_overlaps.shape)
             print '~~~~~~~~~~~~~~~gt_max_overlaps: {}'.format(gt_max_overlaps)

        # 找到所有overlaps中所有等于gt_max_overlaps的元素，因为gt_max_overlaps对于每个非负类别只保留一个
        # anchor，如果同一列有多个相等的最大IOU overlap值，那么就需要把其他的几个值找到，并在后面将它们
        # 的label设为1，即认为它们是object，毕竟在RPN的cls任务中，只要认为它是否是个object即可，即一个
        # 二分类问题。
        gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]
        if DEBUG :
            print '~~~~~~~~~~~~~~~gt_argmax_overlaps: {}'.format(gt_argmax_overlaps)

        if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            # assign bg labels first so that positive labels can clobber them
            labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

        # fg label: for each gt, anchor with highest overlap
        labels[gt_argmax_overlaps] = 1

        # fg label: above threshold IOU
        labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1

        if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            # assign bg labels last so that negative labels can clobber positives
            labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

        # subsample positive labels if we have too many
        num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)
        fg_inds = np.where(labels == 1)[0]
        if len(fg_inds) > num_fg:
            disable_inds = npr.choice(
                fg_inds, size=(len(fg_inds) - num_fg), replace=False)
            labels[disable_inds] = -1

        # subsample negative labels if we have too many
        num_bg = cfg.TRAIN.RPN_BATCHSIZE - np.sum(labels == 1)
        bg_inds = np.where(labels == 0)[0]
        if len(bg_inds) > num_bg:
            disable_inds = npr.choice(
                bg_inds, size=(len(bg_inds) - num_bg), replace=False)
            labels[disable_inds] = -1
            #print "was %s inds, disabling %s, now %s inds" % (
                #len(bg_inds), len(disable_inds), np.sum(labels == 0))

        bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)
        # 求bbox的回归目标
        bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :])

        bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
        # 只对正样本的bbox_inside_weights赋值cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS 为什么？？？？？
        bbox_inside_weights[labels == 1, :] = np.array(cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS)

        bbox_outside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
        if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
            # uniform weighting of examples (given non-uniform sampling)
            num_examples = np.sum(labels >= 0)
            positive_weights = np.ones((1, 4)) * 1.0 / num_examples
            negative_weights = np.ones((1, 4)) * 1.0 / num_examples
        else:
            assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
                    (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))
            positive_weights = (cfg.TRAIN.RPN_POSITIVE_WEIGHT /
                                np.sum(labels == 1))
            negative_weights = ((1.0 - cfg.TRAIN.RPN_POSITIVE_WEIGHT) /
                                np.sum(labels == 0))
        # 正负样本都设置bbox_inside_weights
        bbox_outside_weights[labels == 1, :] = positive_weights
        bbox_outside_weights[labels == 0, :] = negative_weights

        if DEBUG:
            # 计算正样本的box的均值和std
            self._sums += bbox_targets[labels == 1, :].sum(axis=0)
            self._squared_sums += (bbox_targets[labels == 1, :] ** 2).sum(axis=0)
            self._counts += np.sum(labels == 1)
            means = self._sums / self._counts
            stds = np.sqrt(self._squared_sums / self._counts - means ** 2)
            print 'means:'
            print means
            print 'stdevs:'
            print stds

        # map up to original set of anchors 主要是将长度为len(inds_inside)的数据映射回长度
        # total_anchors的数据，total_anchors ～ 2496×9
        labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
        bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
        bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)
        bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)

        if DEBUG:
            print 'rpn: max max_overlap', np.max(max_overlaps)
            print 'rpn: num_positive', np.sum(labels == 1)
            print 'rpn: num_negative', np.sum(labels == 0)
            self._fg_sum += np.sum(labels == 1)
            self._bg_sum += np.sum(labels == 0)
            self._count += 1
            print 'rpn: num_positive avg', self._fg_sum / self._count
            print 'rpn: num_negative avg', self._bg_sum / self._count

        # labels
        labels = labels.reshape((1, height, width, A)).transpose(0, 3, 1, 2)
        labels = labels.reshape((1, 1, A * height, width))
        top[0].reshape(*labels.shape)
        top[0].data[...] = labels

        # bbox_targets
        bbox_targets = bbox_targets \
            .reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)
        top[1].reshape(*bbox_targets.shape)
        top[1].data[...] = bbox_targets

        # bbox_inside_weights
        bbox_inside_weights = bbox_inside_weights \
            .reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)
        assert bbox_inside_weights.shape[2] == height
        assert bbox_inside_weights.shape[3] == width
        top[2].reshape(*bbox_inside_weights.shape)
        top[2].data[...] = bbox_inside_weights

        # bbox_outside_weights
        bbox_outside_weights = bbox_outside_weights \
            .reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)
        assert bbox_outside_weights.shape[2] == height
        assert bbox_outside_weights.shape[3] == width
        top[3].reshape(*bbox_outside_weights.shape)
        top[3].data[...] = bbox_outside_weights

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        # 之前有个疑问：在输出日志里，rpn-data layer 明明是need backward 的，同时rpn_cls_score 这个
        # bottom blob 所对应的 propagatedown 为true ，为什么在anchor_target_layer的定义里没有
        # backward的实现，只是一个pass，仔细想想，其实很简单：在forward方法里，rpn_cls_score的作用只是获取
        # Height, Weight，并没有其他计算，所以does not need propagate gradients, 可见layer的设计要
        # 考虑具体情况
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


def _unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if len(data.shape) == 1:
        ret = np.empty((count, ), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count, ) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret


def _compute_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 5

    return bbox_transform(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)
--------------------- 
作者：iamzhangzhuping 
来源：CSDN 
原文：https://blog.csdn.net/iamzhangzhuping/article/details/51434019 
版权声明：本文为博主原创文章，转载请附上博文链接！