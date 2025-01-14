class RadioInput {
  constructor(name, onChange) {
    this.inputs = document.querySelectorAll(`input[name=${name}]`);
    for (let input of this.inputs) {
      input.addEventListener('change', onChange);
    }
  }

  get value() {
    for (let input of this.inputs) {
      if (input.checked) {
        return input.value;
      }
    }
  }
}

class Input {
  constructor(id, onChange) {
    this.input = document.getElementById(id);
    this.input.addEventListener('change', onChange);
    this.valueAttrib = this.input.type === 'checkbox' ? 'checked' : 'value';
  }

  get value() {
    return this.input[this.valueAttrib];
  }
}

class CubeFace {
  constructor(faceName) {
    this.faceName = faceName;

    this.anchor = document.createElement('a');
    this.anchor.style.position='absolute';
    this.anchor.title = faceName;

    this.img = document.createElement('img');
    this.img.style.filter = 'blur(4px)';

    this.anchor.appendChild(this.img);
  }

  setPreview(url, x, y) {
    this.img.src = url;
    this.anchor.style.left = `${x}px`;
    this.anchor.style.top = `${y}px`;
  }

  setDownload(url, fileExtension) {
    // this.anchor.href = url;
    // this.anchor.download = `${this.faceName}.${fileExtension}`;
    this.img.style.filter = '';
  }
}

class HXImage {
  constructor(url, fileExtension) {
    this.anchor = document.createElement('a');
    this.anchor.style.position='absolute';
    this.anchor.title = 'h-cross';
    this.anchor.href = url;
    this.anchor.download = `${this.faceName}.${fileExtension}`;

    this.img = document.createElement('img');
    this.img.src = url;

    this.anchor.appendChild(this.img);

    this.anchor.style.left = 0;
    this.anchor.style.top = 0;
    this.anchor.style.width = '100%';
    this.anchor.style.height = '100%';
    this.img.style.width = '100%';
    this.img.style.height = '100%';
  }
}

function removeChildren(node) {
  while (node.firstChild) {
    node.removeChild(node.firstChild);
  }
}

const mimeType = {
  'jpg': 'image/jpeg',
  'png': 'image/png'
};

function getDataURL(imgData, extension) {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  canvas.width = imgData.width;
  canvas.height = imgData.height;
  ctx.putImageData(imgData, 0, 0);
  return new Promise(resolve => {
    canvas.toBlob(blob => {
      const url = URL.createObjectURL(blob);
      objectUrls.push(url);
      resolve(url);
    }, mimeType[extension], 0.92);
  });
}

const dom = {
  imageInput: document.getElementById('imageInput'),
  faces: document.getElementById('faces'),
  generating: document.getElementById('generating')
};

dom.imageInput.addEventListener('change', loadImage);

const settings = {
  pitch: new Input('cube-rotation-pitch', loadImage),
  yaw: new Input('cube-rotation-yaw', loadImage),
  roll: new Input('cube-rotation-roll', loadImage),
  interpolation: new RadioInput('interpolation', loadImage),
  format: new RadioInput('format', loadImage),
};

const facePositions = {
  pz: {x: 1, y: 1},
  nz: {x: 3, y: 1},
  px: {x: 2, y: 1},
  nx: {x: 0, y: 1},
  py: {x: 1, y: 0},
  ny: {x: 1, y: 2}
};

function loadImage() {
  const file = dom.imageInput.files[0];

  if (!file) {
    return;
  }

  const img = new Image();

  img.src = URL.createObjectURL(file);

  img.addEventListener('load', () => {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const {width, height} = img;
    canvas.width = width;
    canvas.height = height;
    ctx.drawImage(img, 0, 0);
    const data = ctx.getImageData(0, 0, width, height);

    processImage(data);
    URL.revokeObjectURL(img.src);
  });
}

let finished = 0;
let workers = [];
let objectUrls = [];
let images = [];

function clearObjectUrls() {
  objectUrls.forEach(x => URL.revokeObjectURL(x));
  objectUrls = [];
}

function processImage(data) {
  removeChildren(dom.faces);
  clearObjectUrls();
  images = [];
  dom.generating.style.visibility = 'visible';

  for (let worker of workers) {
    worker.terminate();
  }

  for (let [faceName, position] of Object.entries(facePositions)) {
    renderFace(data, faceName, position, images);
  }
}

function renderFace(data, faceName, position, savedImages) {
  const face = new CubeFace(faceName);
  dom.faces.appendChild(face.anchor);

  const toRad = x => Math.PI * x / 180;
  const options = {
    data: data,
    face: faceName,
    rotation: { pitch: toRad(settings.pitch.value), yaw: toRad(settings.yaw.value), roll: toRad(settings.roll.value) },
    interpolation: settings.interpolation.value,
  };

  // https://stackoverflow.com/questions/21408510/chrome-cant-load-web-worker
  // const worker = new Worker('convert.js');
  const workerBlobUrl = URL.createObjectURL(new Blob(["("+workerScope.toString()+")()"], {type: 'text/javascript'}));
  objectUrls.push(workerBlobUrl);
  const worker = new Worker(workerBlobUrl);

  const setDownload = ({data: imageData}) => {
    const extension = settings.format.value;

    getDataURL(imageData, extension)
      .then(url => face.setDownload(url, extension));
    savedImages.push({ ...position, image: imageData });
    finished++;

    if (finished === 6) {
      finished = 0;
      workers = [];
      generateHX(savedImages, extension);
    }
  };

  const setPreview = ({data: imageData}) => {
    const x = imageData.width * position.x;
    const y = imageData.height * position.y;

    getDataURL(imageData, 'jpg')
      .then(url => face.setPreview(url, x, y));

    worker.onmessage = setDownload;
    worker.postMessage(options);
  };

  worker.onmessage = setPreview;
  worker.postMessage(Object.assign({}, options, {
    maxWidth: 200,
    interpolation: 'linear',
  }));

  workers.push(worker);
}

/** 
 * @param {{ image: ImageData, x: number, y: number }[]} datas 
 * @param {string} extension
 */
 function generateHX(datas, extension) {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  const faceSize = datas[0].image.width;
  canvas.width = faceSize * 4;
  canvas.height = faceSize * 3;
  for(const { image, x, y } of datas) {
    ctx.putImageData(image, x * faceSize, y * faceSize);
  }
  canvas.toBlob(blob => {
    removeChildren(dom.faces);
    clearObjectUrls();
    const url = URL.createObjectURL(blob);
    objectUrls.push(url);
    dom.faces.appendChild(new HXImage(url, extension).anchor);
    dom.generating.style.visibility = 'hidden';
  }, mimeType[extension], 0.92);
}