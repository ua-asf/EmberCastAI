#![allow(non_snake_case)]

use dioxus::{
    document::{Style, Title},
    prelude::*,
};

use serde_json::json;

fn main() {
    dioxus::launch(App);
}

pub static USERNAME: GlobalSignal<Option<String>> = GlobalSignal::new(|| None);
pub static PASSWORD: GlobalSignal<Option<String>> = GlobalSignal::new(|| None);
pub static WKT_STRING: GlobalSignal<Option<String>> = GlobalSignal::new(|| None);
pub static OUTPUT_DATA: GlobalSignal<ProcessingState> =
    GlobalSignal::new(|| ProcessingState::Empty);
pub static STATUS_MESSAGE: GlobalSignal<Option<String>> = GlobalSignal::new(|| None);
pub static INDEX: GlobalSignal<usize> = GlobalSignal::new(|| 0);

pub static THROBBER: Asset = asset!("assets/throbber.svg");

pub static API_ENDPOINT: &str = "http://127.0.0.1:8000";

enum ProcessingState {
    Empty,
    Processing(String), // Throbber or placeholder image location
    Processed { before: RgbImage, after: RgbImage },
}

#[component]
pub fn App() -> Element {
    rsx! {
        Title { "EmberCast AI" }
        // Stylesheet
        // Black background with white text
        Style {
            r#"@font-face {{
                font-family: 'Pixel';
                src: url('assets/fonts/pixeloid/PixeloidSans-mLxMm.ttf') format('truetype');
                font-weight: 400;
                font-display: swap;
                font-style: normal;
            }}

            body {{
                background-color: #020202;
                color: #FEFEFE;
                margin: 0;
                font-family: 'Pixel';
                font-size: 16px;
             }}

            button {{
                font-family: 'Pixel';
                font-size: 18px;
                border-radius: 0px;
            }}

            input {{
                font-family: 'Pixel';
                text-align: center;
            }}"#
        }
        div { style: "text-align: center;
            height: 100%;
            width: 100vw;
            display: flex;
            flex-direction: column;
            gap: 20px;
            height: 100%;
            flex: 1,
            margin: 0px 0px;
            height: 100vh;
            display: flex;
            align-items: center;",
            UIinputs {}
            Separator {}
            RenderImage {}
        }
    }
}

#[component]
fn UIinputs() -> Element {
    let mut username_error = use_signal(|| false);
    let mut password_error = use_signal(|| false);
    let mut wkt_string_error = use_signal(|| false);
    let mut button_clickable = use_signal(|| true);

    rsx! {
        div { style: "display: flex; flex-direction: row; flex: 1 0 auto; justify-content: center; padding: 5px; gap: 20px",
            div {
                p { "Earthdata Username" }
                input {
                    style: format!("border-color: {}", if username_error() { "red" } else { "white" }),
                    oninput: move |e| {
                        *USERNAME.write() = Some(e.value().clone());
                        username_error.set(false);
                    },
                    value: USERNAME(),
                }
            }

            div {
                p { "Earthdata Password" }
                input {
                    style: format!(
                        "letter-spacing: 2pt; border-color: {}",
                        if password_error() { "red" } else { "white" },
                    ),
                    r#type: "password",
                    oninput: move |e| {
                        *PASSWORD.write() = Some(e.value().clone());
                        password_error.set(false);
                    },
                    value: PASSWORD(),
                }
            }

            div {
                p { "WKT String" }
                input {
                    style: format!("border-color: {}", if wkt_string_error() { "red" } else { "white" }),
                    oninput: move |e| {
                        *WKT_STRING.write() = Some(e.value().clone());
                        wkt_string_error.set(false);
                    },
                    value: WKT_STRING(),
                }
            }

            div { style: "margin-top: 20px; display: flex; justify-content: center;",
                button {
                    style: "width: 100%; padding: 5px;",
                    onclick: move |_| {
                        let mut errors = false;
                        if USERNAME.read().clone().is_none_or(|v| v.is_empty()) {
                            username_error.set(true);
                            errors = true;
                        }
                        if PASSWORD.read().clone().is_none_or(|v| v.is_empty()) {
                            password_error.set(true);
                            errors = true;
                        }
                        if WKT_STRING.read().clone().is_none_or(|v| v.is_empty()) {
                            wkt_string_error.set(true);
                            errors = true;
                        }
                        if errors {
                            return;
                        }
                        button_clickable.set(false);
                        let date_format_str = "%Y-%m-%dT%H:%M:%S";
                        let formatted_date: String = chrono::Local::now()
                            .format(date_format_str)
                            .to_string();
                        println!("Formatted date: {}", formatted_date);
                        *OUTPUT_DATA.write() = ProcessingState::Processing(THROBBER.to_string());
                        spawn(async move {
                            run_model(
                                    &USERNAME().unwrap_or_default(),
                                    &PASSWORD().unwrap_or_default(),
                                    &WKT_STRING().unwrap_or_default(),
                                    &formatted_date,
                                )
                                .await;
                            button_clickable.set(true);
                        });
                    },
                    disabled: !button_clickable(),
                    if button_clickable() {
                        "Run Model"
                    } else {
                        "Loading..."
                    }
                }
            }
        }
    }
}

#[component]
fn Separator() -> Element {
    rsx! {
        div { style: "width: 100%; height: 5px; background-color: #FFF;" }
    }
}

#[component]
fn RenderImage() -> Element {
    rsx! {
        div { style: "display: flex; flex-direction: column; justify-content: start; align-items: center; height: 100vh; width: 100vw; overflow: hidden",
            // Render the selected image if any are available
            match *OUTPUT_DATA.read() {
                ProcessingState::Empty => {
                    rsx! { p { style: "font-size: 24px;", "No image available" } }
                },
                ProcessingState::Processing(ref img_path) => {
                    rsx! {
                        img {
                            style: "padding-top: 30px; padding-bottom: 10px; align-self: center;",
                            fill: "#fff",
                            width: "200",
                            height: "200",
                            src: "{img_path}",
                        }
                        p { style: "font-size: 24px;",
                            if let Some(message) = STATUS_MESSAGE.read().clone() {
                                "{message}"
                            } else {
                                "Processing..."
                            }
                        }
                    }
                },
                ProcessingState::Processed { ref before, ref after } => {
                    rsx! {
                        div { style: "display: flex; flex-direction: row; gap: 20px; justify-content: center; align-items: center; padding-top: 10px; padding-bottom: 10px; width: 100%; height: 100%",
                            div { style: "display: flex; flex-direction: column; gap: 10px; justify-content: center; align-items: center; width: 40vw; height: 100;",
                                p { style: "color: red; font-size: 20px", "Before" }
                                RgbImageToBase64 { img: before.clone(), border_color: "red" }
                            }
                            div { style: "display: flex; flex-direction: column; gap: 10px; justify-content: center; align-items: center; width: 40vw; height: 100%",
                                p { style: "color: green; font-size: 20px", "After" }
                                RgbImageToBase64 { img: after.clone(), border_color: "green" }
                            }
                        }
                    }
                }
            }
        }
    }
}

use base64::{Engine as _, engine::general_purpose::STANDARD};
use image::{Rgb, RgbImage};
use std::io::Cursor;

#[component]
fn RgbImageToBase64(img: RgbImage, border_color: &'static str) -> Element {
    let mut buf = Cursor::new(Vec::new());
    img.write_to(&mut buf, image::ImageFormat::Png).unwrap();

    let b64 = STANDARD.encode(buf.into_inner());

    let data_url = format!("data:image/png;base64,{}", b64);

    rsx! {
        img {
            style: "border: 2px solid {border_color}; width: 100%; height: 100%, object-fit: contain; display: block;",
            src: "{data_url}",
            alt: "Brightness map",
        }
    }
}

/// Runs the model using the provided parameters.
/// This function spawns a new process using the `nix run` command.
///
/// # Arguments
/// * `username` - The Earthdata username.
/// * `password` - The Earthdata password.
/// * `wkt_string` - The WKT string for the area of interest.
/// * `date` - The date string in the format "YYYY-MM-DDTHH:MM:SS.mmm".
///
/// # Returns
/// This function does not return a value. It spawns a process and waits for it to finish.
/// The output file path is stored in the `OUTPUT_FILE` global signal.
async fn run_model(username: &str, password: &str, wkt_string: &str, date: &str) {
    // Strip all of the possible prefix combinations from the WKT string
    let wkt_prefix_stripped = strip_prefixes(wkt_string, &["POLYGON", " ", "((", " "]);

    let wkt_stripped = strip_suffixes(wkt_prefix_stripped, &[" ", "))"]);

    println!("Stripped WKT: {}", wkt_stripped);

    // Parse coordinates into actual numbers, not strings
    let wkt_points: Vec<Vec<[f64; 2]>> = vec![
        wkt_stripped
            .split(", ")
            .map(|v| {
                let (x, y) = v.split_once(" ").unwrap();
                [x.parse::<f64>().unwrap(), y.parse::<f64>().unwrap()]
            })
            .collect(),
    ];

    // Use serde_json for proper structure
    let data = json!({
        "username": username,
        "password": password,
        "wkt_points": wkt_points,
        "date_str": date,
    });

    let client = reqwest::Client::new();

    let response = match client
        .post(format!("{}/process_wkt", API_ENDPOINT))
        .json(&data)
        .send()
        .await
    {
        Ok(resp) => resp,
        Err(e) => {
            *STATUS_MESSAGE.write() = Some(format!("Error: Failed to send request - {}", e));
            return;
        }
    };

    if response.status() != 200 {
        *STATUS_MESSAGE.write() = Some(format!(
            "Error: Received status code {}, {}",
            response.status(),
            response.text().await.unwrap_or_default()
        ));
        return;
    }

    // Reset status message
    *STATUS_MESSAGE.write() = None;

    let result = response.text().await.unwrap_or_default();

    // Process result into two IntImage structs
    // Turn response into json

    let json = serde_json::from_str::<serde_json::Value>(&result).unwrap_or_default();

    let original = json["original"]
        .as_array()
        .unwrap_or(&vec![])
        .iter()
        .map(|v| v.as_u64().unwrap_or(0) as u8)
        .collect::<Vec<u8>>();

    let results = json["results"]
        .as_array()
        .unwrap_or(&vec![])
        .iter()
        .map(|v| v.as_u64().unwrap_or(0) as u8)
        .collect::<Vec<u8>>();

    let dem = json["dem"]
        .as_array()
        .unwrap_or(&vec![])
        .iter()
        // Darken the DEM a bit for better visibility
        .map(|v| v.as_u64().unwrap_or(0) as u8 / 2)
        .collect::<Vec<u8>>();

    let dimensions = json["dims"]
        .as_array()
        .unwrap_or(&vec![])
        .iter()
        .map(|v| v.as_u64().unwrap_or(0) as usize)
        .collect::<Vec<usize>>();

    println!("Dimensions: {:?}", dimensions);
    println!("Original length: {}", original.len());
    println!("Results length: {}", results.len());
    println!("DEM length: {}", dem.len());

    let dem_pixels = dem.iter().flat_map(|v| [*v, *v, *v]).collect::<Vec<u8>>();

    let dem_image =
        RgbImage::from_vec(dimensions[0] as u32, dimensions[1] as u32, dem_pixels).unwrap();

    // The before image will be the dem with the original image overlaid on top
    let mut before_final = dem_image.clone();

    let before_rgb = original
        .iter()
        .flat_map(|v| {
            let pixel_brightness = *v as f32 / 255.0;
            let r = 255.0 * pixel_brightness;
            [r as u8, 0, 0]
        })
        .collect::<Vec<u8>>();

    overlay_non_black(
        &mut before_final,
        &RgbImage::from_vec(
            dimensions[0] as u32,
            dimensions[1] as u32,
            before_rgb.clone(),
        )
        .unwrap(),
    );

    let mut after_final = dem_image;

    let after_rgb = results
        .iter()
        .flat_map(|v| {
            let pixel_brightness = *v as f32 / 255.0;
            let p = 255.0 * pixel_brightness;
            [0, p as u8, 0]
        })
        .collect::<Vec<u8>>();

    // Add the model's predictions in red
    overlay_non_black(
        &mut after_final,
        &RgbImage::from_vec(dimensions[0] as u32, dimensions[1] as u32, after_rgb).unwrap(),
    );

    // Add the original data in yellow
    overlay_non_black(
        &mut after_final,
        &RgbImage::from_vec(dimensions[0] as u32, dimensions[1] as u32, before_rgb).unwrap(),
    );

    *OUTPUT_DATA.write() = ProcessingState::Processed {
        before: before_final,
        after: after_final,
    };
}

/// Overlays non-black pixels from `src` onto `dst`.
/// Both images must have the same dimensions.
fn overlay_non_black(dst: &mut RgbImage, src: &RgbImage) {
    assert_eq!(
        dst.dimensions(),
        src.dimensions(),
        "Images must be same size"
    );

    for (x, y, &pixel) in src.enumerate_pixels() {
        if pixel != Rgb([0, 0, 0]) {
            // Get source pixel data
            let mut new_pixel = pixel;

            let dst_pixel = dst.get_pixel(x, y);

            new_pixel[0] = (new_pixel[0] as u16 + dst_pixel[0] as u16).min(255) as u8;
            new_pixel[1] = (new_pixel[1] as u16 + dst_pixel[1] as u16).min(255) as u8;
            new_pixel[2] = (new_pixel[2] as u16 + dst_pixel[2] as u16).min(255) as u8;

            dst.put_pixel(x, y, new_pixel);
        }
    }
}

fn strip_prefixes<'a>(string: &'a str, prefixes: &'a [&'a str]) -> &'a str {
    let mut result = string;
    for prefix in prefixes {
        if result.starts_with(prefix) {
            result = &result[prefix.len()..];
        }
    }
    result
}

fn strip_suffixes<'a>(string: &'a str, suffixes: &'a [&'a str]) -> &'a str {
    let mut result = string;
    for suffix in suffixes {
        if result.ends_with(suffix) {
            result = &result[..result.len() - suffix.len()];
        }
    }
    result
}
