  
  // const myStream = window.open('','','');
const vid = document.getElementById('video');
const testVid = document.getElementById('test');
const canvas = document.querySelector('canvas');
const context = canvas.getContext('2d');

vid.onplay = function() {
  console.log('it is play');
  // const myImg = document.querySelector('img');

  const stream = vid.captureStream();
  
  //myImg.src = frame; 
  //testVid.srcObject = stream;

  canvas.height = vid.videoHeight;
  canvas.width = vid.videoWidth;

  vid.addEventListener('playing', 
    {
      'Interval': setInterval(
        vid.onloadedmetadata = function(e) {
          context.drawImage(vid,0,0,canvas.width,canvas.height);
        }, 100)
    }
  )
};

