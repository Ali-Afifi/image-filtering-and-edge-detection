let upload1 = document.getElementById("image-input1");
let upload2 = document.getElementById("image-input2");

let image1 = document.getElementById("image1-container");
let image2 = document.getElementById("image2-container");

let outputImage = document.getElementById("output-image");

let secondaryImageInput = document.getElementById("secondary-image-input");

let resultBtn = document.getElementById("result-btn");

let selectBox = document.getElementById("select");
let secondarySelectBox = document.getElementById("sub-select");

secondaryImageInput.style.display = "none";
secondarySelectBox.style.display = "none";

selectBox.addEventListener("change", (e) => {
	option = e.target.value;

	if (option == "4") {
		secondarySelectBox.style.display = "block";
		secondarySelectBox.innerHTML = "";

		for (let i = 0; i < 2; i++) {
			var opt = document.createElement("option");
			opt.value = i + 1;

			switch (i) {
				case 0:
					opt.innerHTML = "Histogram";
					break;

				case 1:
					opt.innerHTML = "Distribution Curve";
					break;

				default:
					break;
			}

			secondarySelectBox.appendChild(opt);
		}
	} else if (option == "7") {
		secondarySelectBox.style.display = "block";
		secondarySelectBox.style.display = "block";
		secondarySelectBox.innerHTML = "";

		for (let i = 0; i < 2; i++) {
			var opt = document.createElement("option");
			opt.value = i + 1;

			switch (i) {
				case 0:
					opt.innerHTML = "local threshold";
					break;

				case 1:
					opt.innerHTML = "global threshold";
					break;

				default:
					break;
			}

			secondarySelectBox.appendChild(opt);
		}
	} else {
		secondarySelectBox.style.display = "none";
	}
});

upload1.addEventListener("change", (e) => {
	if (e.target.files.length) {
		const reader = new FileReader();
		reader.onload = (e) => {
			if (e.target.result) {
				let img = document.createElement("img");
				img.id = "image1";
				img.src = e.target.result;

				image1.innerHTML = "";
				image1.appendChild(img);

				const viewer = new Viewer(document.getElementById("image1"), {
					inline: true,
					viewed() {
						viewer.zoomTo(1);
					},
				});
			}
		};
		reader.readAsDataURL(e.target.files[0]);
	}
});

upload2.addEventListener("change", (e) => {
	if (e.target.files.length) {
		const reader = new FileReader();
		reader.onload = (e) => {
			if (e.target.result) {
				let img = document.createElement("img");
				img.id = "image2";
				img.src = e.target.result;

				image2.innerHTML = "";
				image2.appendChild(img);

				const viewer = new Viewer(document.getElementById("image2"), {
					inline: true,
					viewed() {
						viewer.zoomTo(1);
					},
				});
			}
		};
		reader.readAsDataURL(e.target.files[0]);
	}
});

resultBtn.onclick = (e) => {
	let imgDataURL1 = document.getElementById("image1").src;
	// let imgDataURL2 = document.getElementById("image2").src;

	fetch(`${window.location}process`, {
		method: "POST",
		headers: {
			"Content-Type": "application/json",
		},
		body: JSON.stringify({
			image1: imgDataURL1,
			// image2: imgDataURL2 ? imgDataURL2 : "",
			option1: Number(selectBox.value),
			option2:
				selectBox.value == "4" || selectBox.value == "7"
					? Number(secondarySelectBox.value)
					: 0,
		}),
	})
		.then((res) => res.json())
		.then((data) => {
			let img = document.createElement("img");
			img.id = "result-image";
			img.src = data["img"];

			outputImage.innerHTML = "";
			outputImage.appendChild(img);

			const viewer = new Viewer(document.getElementById("result-image"), {
				inline: true,
				viewed() {
					viewer.zoomTo(1);
				},
			});
			// console.log(data);
		})
		.catch((err) => console.log(err));
};
