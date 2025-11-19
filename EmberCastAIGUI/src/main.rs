#![allow(non_snake_case)]

use dioxus::{
    document::{Style, Title},
    prelude::*,
};

use serde_json::json;

fn main() {
    dioxus::launch(App);
}

static USERNAME: GlobalSignal<Option<String>> = GlobalSignal::new(|| None);
static PASSWORD: GlobalSignal<Option<String>> = GlobalSignal::new(|| None);
static WKT_STRING: GlobalSignal<Option<String>> = GlobalSignal::new(|| None);
static OPENTOPO_KEY: GlobalSignal<Option<String>> = GlobalSignal::new(|| None);
static OUTPUT_DATA: GlobalSignal<ProcessingState> = GlobalSignal::new(|| ProcessingState::Empty);
static STATUS_MESSAGE: GlobalSignal<Option<String>> = GlobalSignal::new(|| None);

static THROBBER: Asset = asset!("assets/throbber.svg");
static FONT: Asset = asset!("assets/PixeloidSans.ttf");

static API_ENDPOINT: &str = "http://127.0.0.1:8000";

enum ProcessingState {
    Empty,
    Processing(String), // Throbber or placeholder image location
    Error,
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
                src: url({FONT}) format('truetype');
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

#[derive(PartialEq)]
enum OpenTopoKeyState {
    None,
    UserProvided,
}

#[component]
fn UIinputs() -> Element {
    let mut username_error = use_signal(|| false);
    let mut password_error = use_signal(|| false);
    let mut wkt_string_error = use_signal(|| false);
    let mut button_clickable = use_signal(|| true);
    let mut topo_key_state = use_signal(|| OpenTopoKeyState::None);
    let mut topo_key_error = use_signal(|| false);

    let box_style = "border: 2px solid #FFF; padding: 10px; display: flex; flex-direction: column; justify-content: center; align-items: center;";

    rsx! {
        div { style: "display: flex; flex-direction: row; flex: 1 0 auto; justify-content: center; padding: 5px; padding-bottom: 0px; padding-top: 0px; flex-wrap: wrap;",
            div { style: box_style,
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

            div { style: box_style,
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

            div { style: box_style,
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

            div { style: box_style,
                p { "OpenTopograph API Key" }
                div { style: "display: flex; flex-direction: row; gap: 10px; justify-content: center",
                    select {
                        onchange: move |evt| {
                            let v = evt.value().clone();
                            match &*v {
                                "0" => topo_key_state.set(OpenTopoKeyState::None),
                                "1" => topo_key_state.set(OpenTopoKeyState::UserProvided),
                                _ => {}
                            };
                            topo_key_error.set(false);
                        },
                        option { value: 0, "Use Server Key" }
                        option { value: 1, "Use Personal Key" }
                    }
                    match &*topo_key_state.read() {
                        OpenTopoKeyState::None => {
                            rsx! {}
                        }
                        OpenTopoKeyState::UserProvided => {
                            rsx! {
                                input {
                                    style: format!("border-color: {}", if topo_key_error() { "red" } else { "white" }),
                                    r#type: "password",
                                    oninput: move |e| {
                                        *OPENTOPO_KEY.write() = Some(e.value().clone());
                                        topo_key_error.set(false);
                                    },
                                    value: OPENTOPO_KEY(),
                                }
                            }
                        }
                    }
                }
            }

            div { style: box_style,
                button {
                    style: "width: 100%; height: 100%; padding: 10px;",
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
                        if *topo_key_state.read() == OpenTopoKeyState::UserProvided
                            && OPENTOPO_KEY.read().clone().is_none_or(|v| v.is_empty())
                        {
                            topo_key_error.set(true);
                            errors = true;
                        }

                        if errors {
                            return;
                        }

                        let open_topo_key = match &*topo_key_state.read() {
                            OpenTopoKeyState::None => None,
                            OpenTopoKeyState::UserProvided => OPENTOPO_KEY.read().clone(),
                        };

                        button_clickable.set(false);
                        *STATUS_MESSAGE.write() = None;
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
                                    open_topo_key,
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

use futures_util::StreamExt;

#[component]
fn Separator() -> Element {
    rsx! {
        div { style: "width: 100%; height: 5px; background-color: #FFF;" }
    }
}

#[component]
fn RenderImage() -> Element {
    rsx! {
        div { style: "display: flex; flex-direction: column; justify-content: center; align-items: center; height: 100vh; width: 100vw; overflow: hidden",
            // Render the selected image if any are available
            match *OUTPUT_DATA.read() {
                ProcessingState::Empty => {
                    rsx! {
                        p { style: "font-size: 24px;", "No image available" }
                    }
                }
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
                }
                ProcessingState::Error => {
                    rsx! {
                        p { style: "font-size: 24px; color: red;",
                            if let Some(message) = STATUS_MESSAGE.read().clone() {
                                "{message}"
                            } else {
                                "An unknown error occurred."
                            }
                        }
                    }
                }
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

/// Handles a streamed response from the server.
/// This function reads the streamed response and processes it chunk by chunk.
/// If a returned message starts with `text:` it writes the text to the status message.
/// If a returned message starts with `image:` it returns that data as a `Some(String)`.
/// If a returned message starts with `error:` it sets the status message to the error and returns
/// `None`.
async fn handle_streamed_response(resp: reqwest::Response) -> Option<String> {
    if resp.status() != 200 {
        *STATUS_MESSAGE.write() = Some(format!(
            "Error: Received status code {}, {}",
            resp.status(),
            resp.text().await.unwrap_or_default()
        ));
        *OUTPUT_DATA.write() = ProcessingState::Error;
        return None;
    }

    let mut stream = resp.bytes_stream();

    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(data) => {
                let data = String::from_utf8_lossy(&data).to_string();

                let lines = data.split('\n');

                for line in lines {
                    let text = line.strip_prefix("data: ").unwrap_or(&line).to_string();

                    // Check the prefix of the text
                    if text.strip_prefix("text:").is_some() {
                        *STATUS_MESSAGE.write() = Some(text["text:".len()..].to_string());
                    } else if text.strip_prefix("image:").is_some() {
                        let image_data = text["image:".len()..].to_string();
                        return Some(image_data);
                    } else if text.strip_prefix("error:").is_some() {
                        *STATUS_MESSAGE.write() =
                            Some(format!("Error: {}", &text["error:".len()..]));
                        *OUTPUT_DATA.write() = ProcessingState::Error;
                        return None;
                    } else {
                        continue;
                    }

                    wasmtimer::tokio::sleep(std::time::Duration::from_millis(25)).await;
                }
            }
            Err(e) => {
                *STATUS_MESSAGE.write() = Some(format!("Error: Failed to read chunk - {}", e));
                *OUTPUT_DATA.write() = ProcessingState::Error;
                return None;
            }
        }
    }

    *STATUS_MESSAGE.write() = Some("Error: No image data received.".to_string());
    *OUTPUT_DATA.write() = ProcessingState::Error;
    None
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
async fn run_model(
    username: &str,
    password: &str,
    wkt_string: &str,
    date: &str,
    open_topo_key: Option<String>,
) {
    // Strip all of the possible prefix combinations from the WKT string
    let wkt_prefix_stripped = strip_prefixes(wkt_string, &["POLYGON", " ", "((", " "]);

    let wkt_stripped = strip_suffixes(wkt_prefix_stripped, &[" ", "))"]);

    println!("Stripped WKT: {}", wkt_stripped);

    // Parse coordinates into actual numbers, not strings
    let wkt_points: Vec<Vec<[f64; 2]>> = vec![{
        let result: Result<Vec<[f64; 2]>, String> = wkt_stripped
            .split(", ")
            .map(|v| {
                // Split into x and y components
                let (x_str, y_str) = v.split_once(" ").ok_or_else(|| {
                    format!(
                        "Invalid coordinate format: '{}'. Use 'POLYGON((LAT LONG, .. ))'",
                        v
                    )
                })?;

                // Parse x coordinate
                let x = x_str
                    .parse::<f64>()
                    .map_err(|e| format!("Invalid x coordinate '{}': {}", x_str, e))?;

                // Parse y coordinate
                let y = y_str
                    .parse::<f64>()
                    .map_err(|e| format!("Invalid y coordinate '{}': {}", y_str, e))?;

                Ok([x, y])
            })
            .collect();

        match result {
            Ok(coords) => coords,
            Err(e) => {
                *STATUS_MESSAGE.write() = Some(format!("Failed to parse WKT coordinates: {}", e));
                *OUTPUT_DATA.write() = ProcessingState::Error;
                return;
            }
        }
    }];

    // Use serde_json for proper structure
    let mut data = json!({
        "username": username,
        "password": password,
        "wkt_points": wkt_points,
        "date_str": date,
    });

    if let Some(key) = open_topo_key {
        data["opentopo_key"] = json!(key);
    }

    let client = reqwest::Client::new();

    let response = match client
        .post(format!("{}/process_wkt_streaming", API_ENDPOINT))
        .json(&data)
        .send()
        .await
    {
        Ok(resp) => resp,
        Err(e) => {
            *STATUS_MESSAGE.write() = Some(format!("Error: Failed to send request - {}", e));
            *OUTPUT_DATA.write() = ProcessingState::Error;
            return;
        }
    };

    if response.status() != 200 {
        *STATUS_MESSAGE.write() = Some(format!(
            "Error: Received status code {}, {}",
            response.status(),
            response.text().await.unwrap_or_default()
        ));
        *OUTPUT_DATA.write() = ProcessingState::Error;
        return;
    }

    let result = match handle_streamed_response(response).await {
        Some(data) => data,
        None => {
            // Error message already set in handle_streamed_response
            return;
        }
    };
    
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

    let mut before_rgb = original
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

    // Make the original data yellow to distinguish it from the model's predictions
    before_rgb.chunks_exact_mut(3).for_each(|v| {
        v[1] = v[0]; // G
    });

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

            new_pixel[0] = new_pixel[0].saturating_add(dst_pixel[0]);
            new_pixel[1] = new_pixel[1].saturating_add(dst_pixel[1]);
            new_pixel[2] = new_pixel[2].saturating_add(dst_pixel[2]);

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
