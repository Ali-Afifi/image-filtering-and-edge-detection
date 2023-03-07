let upload1 = document.getElementById("image-input1");
let upload2 = document.getElementById("image-input2");

let image1 = document.getElementById("image1-container");
let image2 = document.getElementById("image2-container");

let secondaryImageInput = document.getElementById("secondary-image-input");

let resultBtn = document.getElementById("result-btn");

let selectBox = document.getElementById("select");

secondaryImageInput.style.display = "none";

selectBox.addEventListener("change", (e) => {
	if (e.target.value == "option3") {
		secondaryImageInput.style.display = "block";
	} else {
		secondaryImageInput.style.display = "none";
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

	fetch(`${window.location}process`, {
		method: "POST",
		headers: {
			"Content-Type": "application/json",
		},
		body: JSON.stringify({
			image1: imgDataURL1,
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

			console.log(data);
		})
		.catch((err) => console.log(err));
};
