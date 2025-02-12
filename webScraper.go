// Dylan Maltos
// EmberCastAI - webScraper.go
// Scrapes KMZ files from ftp.wildfire.gov into /kmz_data directory for extractPolygon.py to use
// Last modified: 1/22/25

package main

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"github.com/gocolly/colly"
)

// Function downloadKMZ - Downloads KMZ file from a ftp.wildfire.gov and saves it
func downloadKMZ(url, savePath string) error {
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("failed to download file: %s", resp.Status)
	}

	// Create the file
	out, err := os.Create(savePath)
	if err != nil {
		return err
	}
	defer out.Close()

	// Write the file content
	_, err = io.Copy(out, resp.Body)
	return err
}

// Function main - Scrapes KMZ files from ftp.wildfire.gov
func main() {
	// Root URL of ftp.wilfire.gov to scrape
	baseURL := "https://ftp.wildfire.gov/public/incident_specific_data/"

	// Save KMZ files in /kmz_data directory
	outputDir := "kmz_data"

	// Create a colly collector
	c := colly.NewCollector(
		colly.AllowedDomains("ftp.wildfire.gov"),
		// No depth limit to explore all directories
		colly.MaxDepth(0),
	)

	// Callback for each discovered link
	c.OnHTML("a[href]", func(e *colly.HTMLElement) {
		link := e.Attr("href")

		// Skip parent directory links and non-directory targets
		if link == "../" || link == "/" {
			return
		}

		fullURL := e.Request.AbsoluteURL(link)

		if strings.HasSuffix(link, "/") { // If the link is a directory, visit it
			c.Visit(fullURL)
			// If the link points to a KMZ file, download it
		} else if strings.HasSuffix(link, ".kmz") {
			// Get relative path
			relativePath := strings.TrimPrefix(fullURL, baseURL)
			// Maintain directory structure
			savePath := filepath.Join(outputDir, relativePath)

			// Create needed directories for the save path
			os.MkdirAll(filepath.Dir(savePath), os.ModePerm)

			fmt.Printf("Downloading: %s\n", fullURL)
			err := downloadKMZ(fullURL, savePath)
			if err != nil {
				fmt.Printf("Error downloading %s: %v\n", fullURL, err)
			} else {
				fmt.Printf("Saved: %s\n", savePath)
			}
		}
	})

	// Handle errors during scraping
	c.OnError(func(_ *colly.Response, err error) {
		fmt.Printf("Error: %v\n", err)
	})

	// Start scraping
	fmt.Println("Starting scraper...")
	c.Visit(baseURL)
	fmt.Println("Scraping complete!")
}
