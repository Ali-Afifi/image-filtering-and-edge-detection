let upload = document.getElementById("image-input1");

let image1 = document.getElementById("image1");
let image2 = document.getElementById("image2");

let resultBtn = document.getElementById("result-btn");

let selectBox = document.getElementById("select");


let croppedImage1, croppedImage2;

upload.addEventListener("change", (e) => {
	if (e.target.files.length) {
		const reader = new FileReader();
		reader.onload = (e) => {
			if (e.target.result) {
				let img = document.createElement("img");
				img.id = "image";
				img.src = e.target.result;

				image1.innerHTML = "";
				image1.appendChild(img);

				cropper1 = new Cropper(img, {
					zoomOnWheel: false,
					movable: false,
					guides: false,
				});

				croppedImage1 = cropper1;
			}
		};
		reader.readAsDataURL(e.target.files[0]);
	}
});

// upload2.addEventListener("change", (e) => {
// 	if (e.target.files.length) {
// 		const reader = new FileReader();
// 		reader.onload = (e) => {
// 			if (e.target.result) {
// 				let img = document.createElement("img");
// 				img.id = "image";
// 				img.src = e.target.result;

// 				image2.innerHTML = "";
// 				image2.appendChild(img);

// 				cropper2 = new Cropper(img, {
// 					zoomOnWheel: false,
// 					movable: false,
// 					guides: false,
// 				});

// 				croppedImage2 = cropper2;
// 			}
// 		};
// 		reader.readAsDataURL(e.target.files[0]);
// 	}
// });

resultBtn.onclick = (e) => {
	let b64Image1 = croppedImage1.getCroppedCanvas().toDataURL("image/png");

	fetch(`${window.location}process`, {
		method: "POST",
		headers: {
			"Content-Type": "application/json",
		},
		body: JSON.stringify({
			image1: b64Image1,
			// image2: b64Image2,
			option: selectBox.value,
			// mag: uniformMagnitudeCheckBox.checked,
			// phase: uniformPhaseCheckBox.checked,
		}),
	})
		.then((res) => res.json())
		.then((data) => {
			result.src = data["img"];
			resultSection.style.display = "block";
		})
		.catch((err) => console.log(err));
};
