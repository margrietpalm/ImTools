__author__ = 'mpalm'
import os,math,glob,csv
from multiprocessing import Pool
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import itertools
import future

__FONTPATH__ = '/usr/share/fonts/truetype/freefont/FreeSans.ttf'

def readColorMap(fn):
    """ Read colormap from tab seperated file.

    Returns:
      Dictionary with (r,g,b) tuple for each key
    """
    f = open(fn)
    reader = csv.reader(f,delimiter='\t')
    return {int(row[0]) : tuple(float(row[i]) for i in range(1,4)) for row in reader}


def addColorBar(imname,cmfile,w,h,labels=None,fontcolor=(0,0,0),bgcolor=(255,255,255),fontpath=__FONTPATH__,
                fontsize=24,outname=None,horizontal=False,title=None):
    """ Add colorbar to an image

    Args:
      imname: image filename
      cmfile: file with the colormap
      w: width of the colorbar
      h: height of the colorbar
      labels: labels to put beside the colorbar
      fontcolor: font color (r,g,b)
      bgcolor: background color (r,g,b)
      fontpath: path to font
      fontsize: font size
      outname: name of the new image
    """

    im = Image.open(imname)
    cm = readColorMap(cmfile)
    if horizontal:
        im = _addColorBarHorizontal(im,cm,w,h,labels,fontcolor,bgcolor,fontpath,fontsize,title)
    else:
        im = _addColorBarVertical(im,cm,w,h,labels,fontcolor,bgcolor,fontpath,fontsize)
    if outname is None:
        im.save(imname)
    else:
        im.save(outname)

def _addColorBarVertical(im,cm,w,h,labels=None,fontcolor=(0,0,0),bgcolor=(255,255,255),
                         fontpath=__FONTPATH__,fontsize=24):
    if labels is not None:
        font = ImageFont.truetype(fontpath, fontsize)
        lablen = [len(label) for label in labels]
        lbig = labels[lablen.index(max(lablen))]
        temp = Image.new('RGB', (10, 10))
        draw = ImageDraw.Draw(temp)
        tsize = draw.textsize(str(lbig), font=font)
        h = h-tsize[1]
        w = w-1.1*tsize[0]-10
    nx = int(math.ceil(w))
    ny = int(math.ceil(h))
    newim = Image.new('RGB',(nx,ny),bgcolor)
    dh = h/float(len(cm))
    draw = ImageDraw.Draw(newim)
    y0 = ny-(ny-h)/2.
    for idx,c in cm.items():
        color = tuple(int(255*c[i]) for i in [0,1,2])
        draw.rectangle([(0,y0-idx*dh),(w,y0-(idx+1)*dh)],fill=color,outline=color)
    if labels is not None:
        x = w+0.1*tsize[0]
        for i,label in enumerate(labels):
            y = y0-.5*tsize[1]-h*i/float(len(labels)-1)
            draw.text((x,y),str(label),fill=fontcolor,font=font)
    im.paste(newim,((im.size[0]-nx)-nx/2,(im.size[1]-ny)/2))
    return im


def _addColorBarHorizontal(im,cm,w,h,labels=None,fontcolor=(0,0,0),bgcolor=(255,255,255),
                         fontpath=__FONTPATH__,fontsize=24,title=None):
    nx = int(math.ceil(w))
    ny = int(math.ceil(h))
    oy = 0
    if labels is not None:
        font = ImageFont.truetype(fontpath, fontsize)
        lablen_left = len(labels[0])
        lablen_right = len(labels[-1])
        temp = Image.new('RGB', (10, 10))
        draw = ImageDraw.Draw(temp)
        if lablen_right > lablen_left:
            tsize = draw.textsize(str(labels[-1]), font=font)
        else:
            tsize = draw.textsize(str(labels[0]), font=font)
        h = h-tsize[1]
        w = w-tsize[0]
        oy = tsize[1]
    if title is not None:
        font = ImageFont.truetype(fontpath, fontsize)
        temp = Image.new('RGB', (10, 10))
        draw = ImageDraw.Draw(temp)
        tsize = draw.textsize(str(title), font=font)
        h = h-tsize[1]
    newim = Image.new('RGB',(nx,ny),bgcolor)
    dw = w/float(len(cm))
    draw = ImageDraw.Draw(newim)
    x0 = (nx-w)/2.
    for idx,c in cm.items():
        color = tuple(int(255*c[i]) for i in [0,1,2])
        draw.rectangle([(x0+idx*dw,oy),(x0+(idx+1)*dw,oy+h)],fill=color,outline=color)
    if labels is not None:
        y = 0-0.1*tsize[1]
        for i,label in enumerate(labels):
            tsize = draw.textsize(str(label), font=font)
            x = x0-.5*tsize[0]+w*i/float(len(labels)-1)
            draw.text((x,y),str(label),fill=fontcolor,font=font)
    if title is not None:
        tsize = draw.textsize(str(title), font=font)
        y = oy+h
        x = .5*(nx-tsize[0])
        draw.text((x, y), str(title), fill=fontcolor, font=font)
    im.paste(newim,(int((im.size[0]-nx)/2),int((im.size[1]-ny)-ny/2)))
    return im


def _addMPLColorBarVertical(im,cm,w,h,labels=None,fontcolor=(0,0,0),bgcolor=(255,255,255),fontpath=__FONTPATH__):
    # create new image and paste original image on the left side
    ny = int(h)
    nx = int(w)
    if labels is not None:
        fontsize = 24
        font = ImageFont.truetype(fontpath, fontsize)
        lablen = [len(label) for label in labels]
        lbig = labels[lablen.index(max(lablen))]
        temp = Image.new('RGB', (10, 10))
        draw = ImageDraw.Draw(temp)
        tsize = draw.textsize(str(lbig), font=font)
        h = h-tsize[1]
        w = w-1.1*tsize[0]-10
    newim = Image.new('RGB',(nx,ny),bgcolor)
    # make vertical colorbar
    dh = h/100.0
    draw = ImageDraw.Draw(newim)
    y0 = ny-(ny-h)/2.
    for i,a in enumerate(np.linspace(0,1,100)):
        color = tuple(int(255*cm(a)[idx]) for idx in range(3))
        draw.rectangle([(0,y0-i*dh),(w,y0-(i+1)*dh)],fill=color,outline=color)
    if labels is not None:
        x = w+0.1*tsize[0]
        for i,label in enumerate(labels):
            y = y0-.5*tsize[1]-h*i/float(len(labels)-1)
            draw.text((x,y),str(label),fill=fontcolor,font=font)
    im.paste(newim,(im.size[0]-nx,0))
    return im

def _makeTransparent(im, bgcolor=(255, 255, 255)):
    """ Make background transparent

    :param im: image
    :param bgcolor: color of pixels that will be made transparent
    :return: transparent image
    """
    im = im.convert("RGBA")
    imdata = np.asarray(im, dtype=np.int)
    pos = np.where((imdata[:, :, 0] == bgcolor[0]) & (imdata[:, :, 1] == bgcolor[1]) & (imdata[:, :, 2] == bgcolor[2]))
    imdata[pos] = [bgcolor[0], bgcolor[1], bgcolor[2], 0]
    return Image.fromarray(np.uint8(imdata))


def _addBox(im, bw, bh, pos, color=None):
    """ Draw border around image

    :param im: image
    :param bw: box width
    :param bh: box height
    :param pos: center of the box
    :param color: box color
    :return: image with box
    """
    if color is None:
        color = (0, 0, 0)
    draw = ImageDraw.Draw(im)
    draw.rectangle([pos, (pos[0] + bw, pos[1] + bh)], outline=color)


def _addTimeStamp(im, stamp, fs=6, fc=None, fontpath=__FONTPATH__):

    if fc is None:
        fc = (0, 0, 0)
    draw = ImageDraw.Draw(im)
    fontsize = int(fs)
    font = ImageFont.truetype(fontpath, fontsize)
    (w, h) = draw.textsize(str(stamp), font=font)
    (nx, ny) = im.size
    y = ny - 2 * h
    x = nx - (w + .5 * h)
    draw.text((x, y), str(stamp), fill=fc, font=font)

def _getLabelSize(label, fs=8, fontpath=__FONTPATH__):
    im = Image.new('RGB', (10, 10))
    draw = ImageDraw.Draw(im)
    fontsize = int(fs)
    font = ImageFont.truetype(fontpath, fontsize)
    return draw.textsize(str(label), font=font)

def _addLabel(im, label, fs=8, fc=None, fontpath=__FONTPATH__, top=True):
    """ Draw label at the top center of the image.

    :param im: image
    :param label: label
    :param fs: font size
    :param fc: font color (r,g,b)
    :param fontpath: path to font
    :return: image with label
    """
    if fc is None:
        fc = (0, 0, 0)
    draw = ImageDraw.Draw(im)
    fontsize = int(fs)
    font = ImageFont.truetype(fontpath, fontsize)
    (w, h) = _getLabelSize(label, fs, fontpath)
    (nx, ny) = im.size
    if top:
        y = int(.5 * h)
    else:
        y = int(ny - 1.5 * h)
    x = int(0.5 * (nx - w))

    draw.text((x, y), str(label), fill=fc, font=font)


def stackImageSeries(idmap, outbase, geometry, postfix='', imdir='images/', labelsordered=None, fontsize=20, scale=1,
                     bcolor=(255, 255, 255),fcolor=(0, 0, 0), fontpath=__FONTPATH__, imdist=0, ncores=1, title=None,
                     border=False, imbases=None):
    #print '{}/{}/*.png'.format(imdir,idmap.ke)
    tlist = map(lambda s: s.split('_')[-1].replace('.png',''),glob.glob('{}/{}/*.png'.format(imdir,idmap.keys()[0])))
    imname = lambda simid,t: imdir+simid+'/'+simid+postfix+'_'+t+'.png'
    if imbases is None:
        imbases = {k : k for k in idmap}
    jobs = []
    for t in tlist:
        imdict = {name : imname(simid,t) for simid,name in idmap.items()}
        jobs.append([imdict,geometry,outbase+postfix+'_'+t+'.png',True,title,fontsize,border,scale,
                     bcolor,fcolor,fontpath,labelsordered,imdist])

    if ncores == 1:
        for job in jobs:
            stackImagesMC(job)
    else:
        pool = Pool(processes=ncores)
        pool.map(stackImagesMC,jobs)


def stackImagesMC(args):
    [images,geometry,filename,label,title,fontsize,border,scale,bcolor,fcolor,fontpath,labelsordered,imdist] = args
    stackImages(images,geometry,filename,label,title,fontsize,border,scale,bcolor,fcolor,fontpath,labelsordered,imdist)

def stackImages(images, geometry, filename, label=False, title=None, fontsize=20, border=False, scale=1,
                bcolor=None, fcolor=(0, 0, 0),fontpath=__FONTPATH__, labelsordered=None,imdist=0):
    """ Stack a set of images together in one image.

    :param images: dictionary with labels as keys and image filenames as values
    :param geometry: number of rows and columns (x,y)
    :param filename: target of the stacked image
    :param label: add labels to the subimages
    :param title: overall title for image
    :param fontsize: fontsize for labels and title
    :param border: add border to subimages
    :type border: bool
    :param scale: scaling factor of the created picture
    :param bcolor: background color
    :param fcolor: font color
    :param fontpath: path to the font
    :param labelsordered: list of labels (same as keys of images) in the order that the images should be stacked
    """
    if label or title:
        fontsize = int(fontsize * scale)
        font = ImageFont.truetype(fontpath, fontsize)
    if labelsordered is not None:
        labels = labelsordered
    else:
        labels = images.keys()
        labels.sort()
    if bcolor is None:
        bcolor = Image.open(images.values()[0]).getpixel((0,0))[0:3]
    nx = 0
    ny = 0
    for im in images.values():
        sz = Image.open(im).size
        if sz[0] > nx:
            nx = sz[0]
            ny = sz[1]
    imsize = (scale * nx+imdist, scale * ny+imdist)
    # ---- Create new image ----#
    offsetX = 5
    offsetY = 0
    imw = int(np.ceil(imsize[0] * geometry[0] + 2 * offsetX))
    imh = int(np.ceil(imsize[1] * geometry[1] + offsetY))
    if title is not None:
        im = Image.new('RGB', (10, 10))
        draw = ImageDraw.Draw(im)
        tsize = draw.textsize(str(title), font=font)
        im = Image.new('RGBA', (imw, imh + tsize[1]), bcolor)
        offsetY = tsize[1]
    else:
        im = Image.new('RGBA', (imw, imh), bcolor)
        offsetY = 0
    offsetX += .5*imdist
    offsetY += .5*imdist
    draw = ImageDraw.Draw(im)

    #----- Put plots on canvas ----#
    for i, l in enumerate(labels):
        x0 = int(i % geometry[0] * imsize[0] + offsetX)
        y0 = int((i / geometry[0]) * imsize[1] + offsetY)
        if os.path.isfile(images[l]):
            importim = Image.open(images[l])
            if importim.mode == 'L':
                importim = importim.convert('RGB')
            if label:
                (w, h) = _getLabelSize(labels[i], fontsize)
                subim = Image.new('RGBA', (int(nx * scale), int(ny * scale)+h), bcolor)
                subim.paste(importim.resize((int(nx * scale), int(ny * scale))), (0, 0))
                _addLabel(subim, labels[i], fs=fontsize, top=False, fc=fcolor)
                im.paste(subim, (x0, y0))
            else:
                im.paste(importim.resize((int(nx * scale), int(ny * scale))), (x0, y0))
                #subim = Image.new('RGBA', (int(nx * scale), int(ny * scale))), bcolor)
            #if label:
            #im.paste(newim.resize((int(nx * scale), int(ny * scale))), (x0, y0))
        if border:
            draw.rectangle([(x0, y0), (x0 + int(nx * scale), y0 + int(ny * scale))], outline=fcolor)
    if not (title is None):
        tsize = draw.textsize(str(title), font=font)
        draw.text((im.size[0] / 2.0 - 0.5 * tsize[0] + offsetX, 0), str(title), fill=fcolor, font=font)

    #----- Save image -----#
    if scale is not 1:
        im = im.resize((int(scale * im.size[0]), int(scale * im.size[1])))
    im.save(filename)


def morphImageSeries(idmap, outbase, postfix='', imdir='images/', xlabel=None, ylabel=None, xtics=None, ytics=None, fontsize=20, scale=1, border=False,
                title=None, bcolor=(255, 255, 255),fcolor=(0, 0, 0), fontpath=__FONTPATH__, delta=0, ncores=1, cropsize=None):
    tlist = []
    for imid in list(itertools.chain.from_iterable(idmap)):
        _tlist = map(lambda s: s.split('_')[-1].replace('.png',''),glob.glob('{0}{1}/{1}{2}*.png'.format(imdir,imid,postfix)))
        if len(_tlist) > len(tlist):
            tlist = _tlist
    imname = lambda simid,t: imdir+simid+'/'+simid+postfix+'_'+t+'.png'
    jobs = []
    for t in tlist:
        imarray = [[imname(simid,t) for simid in row] for row in idmap]
        jobs.append([imarray,outbase+postfix+'_'+t+'.png',xlabel,ylabel,xtics,ytics,fontsize,scale,border,title,bcolor,fcolor,
                     fontpath,delta,cropsize])
    if ncores == 1:
        for job in jobs:
            morphImagesMC(job)
    else:
        pool = Pool(processes=ncores)
        pool.map(morphImagesMC,jobs)


def morphImagesMC(args):
    [images,filename,xlabel,ylabel,xtics,ytics,fontsize,scale,border,title,bgcolor,fcolor,fontpath,delta,cropsize] = args
    morphImages(images,filename,xlabel,ylabel,xtics,ytics,fontsize,scale,border,title,bgcolor,fcolor,fontpath,delta,cropsize)

def morphImages(images, filename, xlabel=None, ylabel=None, xtics=None, ytics=None, fontsize=20, scale=1, border=False,
                title=None, bcolor=(255, 255, 255), fcolor=(0, 0, 0), fontpath=__FONTPATH__, delta=0, cropsize=None):
    """ Stack a set of images together in one morphospace.

    :param images: 2D array with image filenames
    :param filename: target of the stacked image
    :param xlabel: label to be plotted on x-axis
    :param ylabel: label to be plotted on y-axis
    :param xtics: items on x-axis
    :param ytics: items on y-axis
    :param fontsize: fontsize for labels and title
    :param scale: scaling factor of the created picture
    :param border: add border to subimages
    :param title: overall title for image
    :param bcolor: background color
    :param fcolor: font color
    :param fontpath: path to the font
    """
    font = ImageFont.truetype(fontpath, fontsize)
    tfont = ImageFont.truetype(fontpath, int(1.1 * fontsize))
    imlist = [images[i][j] for i in range(len(images)) for j in range(len(images[0]))]
    i = 0
    while (not os.path.isfile(imlist[i])) and (i <= len(imlist)):
        i += 1
    if i == len(imlist):
        print('none of the images exists')
        return
    orgsize = Image.open(imlist[i]).size
    if cropsize is None:
        subimsize = (int(scale * orgsize[0] + delta), int(scale * orgsize[1] + delta))
    else:
        dx = (orgsize[0]-cropsize[0])/2
        dy = (orgsize[1]-cropsize[1])/2
        box = (dx,dy,orgsize[0]-dx,orgsize[1]-dy)
        subimsize = (int(scale * cropsize[0] + delta), int(scale * cropsize[1] + delta))
    imsize = [subimsize[0] * len(images[0]) - delta, subimsize[1] * len(images) - delta]
    oX = 0
    oY = 0
    toX = 0
    toY = 0
    if xlabel is not None:
        im = Image.new('RGB', (10, 10))
        draw = ImageDraw.Draw(im)
        lsize = draw.textsize(str(xlabel), font=font)
        imsize[1] = imsize[1] + lsize[1]
        oY += lsize[1]
        toY += lsize[1]
    if ylabel is not None:
        im = Image.new('RGB', (10, 10))
        draw = ImageDraw.Draw(im)
        lsize = draw.textsize(str(ylabel), font=font)
        imsize[0] += lsize[1]
        oX += lsize[1]
        toX += lsize[1]
    if xtics is not None:
        im = Image.new('RGB', (10, 10))
        draw = ImageDraw.Draw(im)
        lsize = draw.textsize(str(xtics[0]), font=font)
        imsize[1] += lsize[1]
        oY += lsize[1]
    if ytics is not None:
        im = Image.new('RGB', (10, 10))
        draw = ImageDraw.Draw(im)
        lsize = draw.textsize(str(ytics[0]), font=font)
        imsize[0] += lsize[1]
        oX += lsize[1]
    loY = 0
    if title is not None:
        im = Image.new('RGB', (10, 10))
        draw = ImageDraw.Draw(im)
        lsize = draw.textsize(str(title), tfont)
        offset = lsize[1] + 20 * scale
        imsize[1] += offset
        oY += offset
        if xlabel is not None:
            loY += offset
        toY += offset
    #---- Create new image ----#
    im = Image.new('RGBA', imsize, bcolor)
    draw = ImageDraw.Draw(im)
    #----- Put plots on canvas ----#
    for i in range(len(images)):
        for j in range(len(images[0])):
            #~ print i,j,len(images),len(images[0])
            if not os.path.isfile(images[i][j]):
                continue
            newim = Image.open(images[i][j])
            x0 = j * subimsize[0] + oX
            y0 = i * subimsize[1] + oY
            #~ print x0,y0,x0+newim.size[0],y0+newim.size[1]
            if cropsize is not None:
                newim = newim.crop(box)
                im.paste(newim.resize((int(scale * cropsize[0]), int(scale * cropsize[1]))), (x0, y0))
            else:
                im.paste(newim.resize((int(scale * orgsize[0]), int(scale * orgsize[1]))), (x0, y0))

            #~ im.paste(newim.resize(subimsize),(x0,y0))
            if border:
                draw.rectangle([(x0, y0), (x0 + newim.size[0], y0 + newim.size[1])], outline=fcolor)
    #----- Draw axis and tics -----#
    ticsfont = ImageFont.truetype(fontpath, int(.75 * fontsize))
    if xtics is not None:
        for i in range(len(xtics)):
            tsize = draw.textsize(str(xtics[i]), font=ticsfont)
            x0 = (i + 0.5) * subimsize[0] + oX - int(tsize[0] / 2.0)
            y0 = toY
            for n, p in enumerate(xtics[i].split('\n')):
                tsize = draw.textsize(str(p), font=ticsfont)
                x0 = (i + 0.5) * subimsize[0] + oX - int(tsize[0] / 2.0)
                #~ y0p = (len(l.split('\n'))-1-n)*tsize[1]+y0
                y0p = y0 + n * tsize[1]
                draw.text((x0, y0p), str(p), fill=fcolor, font=ticsfont)
                #~ print 'xtic at ',(x0,y0)
                #~ draw.text((x0,y0),str(xtics[i]),fill=fcolor,font=ticsfont)
    if ytics is not None:
        for i in range(len(ytics)):
            x0 = toX
            y0 = int((i + .5) * subimsize[1] + oY)
            for n, p in enumerate(ytics[i].split('\n')):
                tsize = draw.textsize(p, font=ticsfont)
                tim = Image.new('RGBA', (tsize[0], tsize[1]), bcolor)
                tdraw = ImageDraw.Draw(tim)
                tdraw.text((0, 0), p, fill=fcolor, font=ticsfont)
                x0p = x0 + n * tsize[1]
                #~ draw.text((x0p,y0),str(p),fill=fcolor,font=ticsfont)
                tim = tim.rotate(90,expand=True)
                y0p = y0 - int(tsize[0] / 2.0)
                im.paste(tim, (x0p, y0p))
                #~ tdraw.text((0,0),str(ytics[i]),fill=fcolor,font=ticsfont)
                #~ tim = tim.rotate(90)
                #~ y0 -= int(tsize[0]/2.0)
                #~ im.paste(tim,(x0,y0))
    if xlabel is not None:
        tsize = draw.textsize(str(xlabel), font=font)
        draw.text((im.size[0] / 2.0 - 0.5 * tsize[0], loY), str(xlabel), fill=fcolor, font=font)
    if ylabel is not None:
        tsize = draw.textsize(str(ylabel), font=font)
        tim = Image.new('RGBA', (tsize[0], tsize[1]), bcolor)
        tdraw = ImageDraw.Draw(tim)
        tdraw.text((0, 0), str(ylabel), fill=fcolor, font=font)
        tim = tim.rotate(90,expand=True)
        im.paste(tim, (0, int(im.size[1] / 2.0 - tsize[0])))
    if title is not None:
        tsize = draw.textsize(str(title), font=tfont)
        draw.text((im.size[0] / 2.0 - 0.5 * tsize[0], 0), str(title), fill=fcolor, font=tfont)
    #----- Save image -----#
    im.save(filename)
