// TropicalCycloneML - Live Dashboard
// Fetches real cyclone data from NOAA and displays on an interactive map

(function () {
    'use strict';

    // --- Configuration ---
    const NOAA_ACTIVE_URL = 'https://www.nhc.noaa.gov/CurrentSummaries.json';
    const NOAA_GIS_URL = 'https://www.nhc.noaa.gov/gis/forecast/archive/';

    // Saffir-Simpson category thresholds (knots)
    const CATEGORIES = [
        { min: 0, max: 33, label: 'TD', cls: 'cat-td', color: '#6c757d' },
        { min: 34, max: 63, label: 'TS', cls: 'cat-ts', color: '#5bc0de' },
        { min: 64, max: 82, label: 'Cat 1', cls: 'cat-1', color: '#ffe066' },
        { min: 83, max: 95, label: 'Cat 2', cls: 'cat-2', color: '#ffa94d' },
        { min: 96, max: 112, label: 'Cat 3', cls: 'cat-3', color: '#ff6b6b' },
        { min: 113, max: 136, label: 'Cat 4', cls: 'cat-4', color: '#e64980' },
        { min: 137, max: 999, label: 'Cat 5', cls: 'cat-5', color: '#be4bdb' },
    ];

    // --- State ---
    let map;
    let stormLayers = [];
    let allStorms = [];

    // --- Map Setup ---
    function initMap() {
        map = L.map('map', {
            center: [20, -60],
            zoom: 3,
            minZoom: 2,
            maxZoom: 12,
            zoomControl: true,
            attributionControl: true,
        });

        // Dark-themed tile layer
        L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> &copy; <a href="https://carto.com/">CARTO</a>',
            subdomains: 'abcd',
            maxZoom: 19,
        }).addTo(map);
    }

    // --- Data Fetching ---
    async function fetchActiveStorms() {
        try {
            // Try NOAA current summaries
            const response = await fetch(NOAA_ACTIVE_URL);
            if (response.ok) {
                const data = await response.json();
                return parseNOAAData(data);
            }
        } catch (e) {
            console.log('NOAA API unavailable, using demo data:', e.message);
        }

        // Fallback: return sample data for demonstration
        return getSampleStorms();
    }

    function parseNOAAData(data) {
        const storms = [];
        if (data && data.activeStorms) {
            data.activeStorms.forEach(function (storm) {
                storms.push({
                    id: storm.id || storm.binNumber,
                    name: storm.name || 'Unnamed',
                    basin: storm.classification || 'NA',
                    lat: parseFloat(storm.latitude) || 0,
                    lon: parseFloat(storm.longitude) || 0,
                    wind: parseInt(storm.intensity) || 0,
                    pressure: parseInt(storm.pressure) || 0,
                    movement: storm.movementDir + ' at ' + storm.movementSpeed + ' mph',
                    type: storm.type || 'tropical',
                    advisory: storm.lastUpdate || '',
                    track: storm.track || [],
                });
            });
        }
        return storms;
    }

    function getSampleStorms() {
        // Realistic sample data to demonstrate the dashboard even when no storms are active
        return [
            {
                id: 'AL052025',
                name: 'EPSILON',
                basin: 'NA',
                lat: 26.4,
                lon: -71.2,
                wind: 85,
                pressure: 974,
                movement: 'NW at 12 mph',
                type: 'HU',
                advisory: 'Advisory 15 - Demo Data',
                track: [
                    { lat: 18.5, lon: -58.3, wind: 35, time: 'Sep 10 00Z' },
                    { lat: 19.8, lon: -60.1, wind: 45, time: 'Sep 10 12Z' },
                    { lat: 21.2, lon: -62.4, wind: 60, time: 'Sep 11 00Z' },
                    { lat: 22.8, lon: -64.8, wind: 75, time: 'Sep 11 12Z' },
                    { lat: 24.5, lon: -67.2, wind: 85, time: 'Sep 12 00Z' },
                    { lat: 26.4, lon: -71.2, wind: 85, time: 'Sep 12 12Z' },
                ],
                forecast: [
                    { lat: 28.1, lon: -73.5, wind: 90, time: '+24h' },
                    { lat: 30.5, lon: -74.0, wind: 80, time: '+48h' },
                    { lat: 33.8, lon: -72.1, wind: 65, time: '+72h' },
                ],
            },
            {
                id: 'AL062025',
                name: 'ZETA',
                basin: 'NA',
                lat: 16.8,
                lon: -42.5,
                wind: 50,
                pressure: 1002,
                movement: 'WNW at 18 mph',
                type: 'TS',
                advisory: 'Advisory 5 - Demo Data',
                track: [
                    { lat: 14.2, lon: -35.0, wind: 30, time: 'Sep 12 00Z' },
                    { lat: 15.0, lon: -37.5, wind: 35, time: 'Sep 12 12Z' },
                    { lat: 15.8, lon: -39.8, wind: 45, time: 'Sep 13 00Z' },
                    { lat: 16.8, lon: -42.5, wind: 50, time: 'Sep 13 12Z' },
                ],
                forecast: [
                    { lat: 17.5, lon: -45.8, wind: 60, time: '+24h' },
                    { lat: 18.2, lon: -49.3, wind: 70, time: '+48h' },
                    { lat: 19.5, lon: -53.0, wind: 75, time: '+72h' },
                ],
            },
            {
                id: 'WP152025',
                name: 'KONG-REY',
                basin: 'WP',
                lat: 22.5,
                lon: 134.8,
                wind: 130,
                pressure: 920,
                movement: 'NW at 8 mph',
                type: 'STY',
                advisory: 'Advisory 22 - Demo Data',
                track: [
                    { lat: 12.0, lon: 142.5, wind: 35, time: 'Sep 08 00Z' },
                    { lat: 13.5, lon: 141.0, wind: 55, time: 'Sep 08 12Z' },
                    { lat: 15.2, lon: 139.8, wind: 75, time: 'Sep 09 00Z' },
                    { lat: 17.0, lon: 138.5, wind: 100, time: 'Sep 09 12Z' },
                    { lat: 19.0, lon: 137.0, wind: 120, time: 'Sep 10 00Z' },
                    { lat: 20.8, lon: 136.0, wind: 130, time: 'Sep 10 12Z' },
                    { lat: 22.5, lon: 134.8, wind: 130, time: 'Sep 11 00Z' },
                ],
                forecast: [
                    { lat: 24.5, lon: 133.0, wind: 120, time: '+24h' },
                    { lat: 27.0, lon: 131.5, wind: 100, time: '+48h' },
                    { lat: 30.5, lon: 131.0, wind: 75, time: '+72h' },
                ],
            },
            {
                id: 'EP102025',
                name: 'NORMA',
                basin: 'EP',
                lat: 19.2,
                lon: -108.5,
                wind: 105,
                pressure: 960,
                movement: 'NNE at 6 mph',
                type: 'HU',
                advisory: 'Advisory 12 - Demo Data',
                track: [
                    { lat: 14.5, lon: -104.0, wind: 30, time: 'Sep 11 00Z' },
                    { lat: 15.5, lon: -105.2, wind: 50, time: 'Sep 11 12Z' },
                    { lat: 16.8, lon: -106.5, wind: 75, time: 'Sep 12 00Z' },
                    { lat: 18.0, lon: -107.5, wind: 95, time: 'Sep 12 12Z' },
                    { lat: 19.2, lon: -108.5, wind: 105, time: 'Sep 13 00Z' },
                ],
                forecast: [
                    { lat: 20.5, lon: -108.8, wind: 100, time: '+24h' },
                    { lat: 22.0, lon: -108.5, wind: 80, time: '+48h' },
                    { lat: 23.5, lon: -107.0, wind: 50, time: '+72h' },
                ],
            },
        ];
    }

    // --- Category Helpers ---
    function getCategory(windKnots) {
        for (var i = CATEGORIES.length - 1; i >= 0; i--) {
            if (windKnots >= CATEGORIES[i].min) return CATEGORIES[i];
        }
        return CATEGORIES[0];
    }

    // --- Map Rendering ---
    function clearMap() {
        stormLayers.forEach(function (layer) {
            map.removeLayer(layer);
        });
        stormLayers = [];
    }

    function renderStorms(storms) {
        clearMap();

        storms.forEach(function (storm) {
            var cat = getCategory(storm.wind);

            // Draw track line if available
            if (storm.track && storm.track.length > 1) {
                var trackPoints = storm.track.map(function (pt) {
                    return [pt.lat, pt.lon];
                });

                // Color segments by intensity
                for (var i = 1; i < storm.track.length; i++) {
                    var segCat = getCategory(storm.track[i].wind);
                    var segment = L.polyline(
                        [
                            [storm.track[i - 1].lat, storm.track[i - 1].lon],
                            [storm.track[i].lat, storm.track[i].lon],
                        ],
                        {
                            color: segCat.color,
                            weight: 3,
                            opacity: 0.8,
                        }
                    ).addTo(map);
                    stormLayers.push(segment);

                    // Small circle at each track point
                    var dot = L.circleMarker(
                        [storm.track[i].lat, storm.track[i].lon],
                        {
                            radius: 3,
                            color: segCat.color,
                            fillColor: segCat.color,
                            fillOpacity: 1,
                            weight: 1,
                        }
                    ).addTo(map);
                    dot.bindPopup(
                        '<strong>' + storm.name + '</strong><br>' +
                        storm.track[i].time + '<br>' +
                        'Wind: ' + storm.track[i].wind + ' kt'
                    );
                    stormLayers.push(dot);
                }
            }

            // Draw forecast track if available
            if (storm.forecast && storm.forecast.length > 0) {
                var forecastPoints = [[storm.lat, storm.lon]];
                storm.forecast.forEach(function (pt) {
                    forecastPoints.push([pt.lat, pt.lon]);
                });

                var forecastLine = L.polyline(forecastPoints, {
                    color: '#ffffff',
                    weight: 2,
                    opacity: 0.5,
                    dashArray: '8, 8',
                }).addTo(map);
                stormLayers.push(forecastLine);

                // Forecast position markers
                storm.forecast.forEach(function (pt) {
                    var fcCat = getCategory(pt.wind);
                    var fcMarker = L.circleMarker([pt.lat, pt.lon], {
                        radius: 5,
                        color: '#ffffff',
                        fillColor: fcCat.color,
                        fillOpacity: 0.6,
                        weight: 1,
                        dashArray: '3, 3',
                    }).addTo(map);
                    fcMarker.bindPopup(
                        '<strong>' + storm.name + ' Forecast</strong><br>' +
                        pt.time + '<br>' +
                        'Wind: ' + pt.wind + ' kt (' + fcCat.label + ')'
                    );
                    stormLayers.push(fcMarker);
                });
            }

            // Current position marker (large, prominent)
            var marker = L.circleMarker([storm.lat, storm.lon], {
                radius: 10 + (storm.wind / 20),
                color: '#ffffff',
                fillColor: cat.color,
                fillOpacity: 0.9,
                weight: 2,
                className: 'storm-marker-active',
            }).addTo(map);

            marker.bindPopup(
                '<strong>' + storm.name + '</strong> (' + storm.id + ')<br>' +
                '<strong>Category:</strong> ' + cat.label + '<br>' +
                '<strong>Wind:</strong> ' + storm.wind + ' kt<br>' +
                '<strong>Pressure:</strong> ' + storm.pressure + ' mb<br>' +
                '<strong>Movement:</strong> ' + storm.movement + '<br>' +
                '<em>' + storm.advisory + '</em>'
            );

            marker.on('click', function () {
                showStormDetail(storm);
            });

            stormLayers.push(marker);

            // Wind radius ring (approximate)
            if (storm.wind >= 64) {
                var radiusKm = storm.wind * 1.5; // rough approximation
                var ring = L.circle([storm.lat, storm.lon], {
                    radius: radiusKm * 1000,
                    color: cat.color,
                    fillColor: cat.color,
                    fillOpacity: 0.08,
                    weight: 1,
                    dashArray: '4, 4',
                }).addTo(map);
                stormLayers.push(ring);
            }
        });
    }

    // --- UI Updates ---
    function updateStormList(storms) {
        var listEl = document.getElementById('active-storms-list');

        if (storms.length === 0) {
            listEl.innerHTML = '<div class="no-storms">No active tropical cyclones.<br>Showing sample data for demonstration.</div>';
            return;
        }

        listEl.innerHTML = '';
        storms.forEach(function (storm) {
            var cat = getCategory(storm.wind);
            var card = document.createElement('div');
            card.className = 'storm-card';
            card.innerHTML =
                '<div class="storm-card-header">' +
                '  <span class="storm-name">' + storm.name + '</span>' +
                '  <span class="storm-category ' + cat.cls + '">' + cat.label + '</span>' +
                '</div>' +
                '<div class="storm-info">' +
                '  <span>Wind: <strong>' + storm.wind + ' kt</strong></span>' +
                '  <span>Pressure: <strong>' + storm.pressure + ' mb</strong></span>' +
                '</div>';

            card.addEventListener('click', function () {
                map.setView([storm.lat, storm.lon], 6);
                showStormDetail(storm);
            });

            listEl.appendChild(card);
        });
    }

    function updateStats(storms) {
        var totalStorms = storms.length;
        var hurricanes = storms.filter(function (s) { return s.wind >= 64; }).length;
        var major = storms.filter(function (s) { return s.wind >= 96; }).length;
        var maxWind = storms.reduce(function (max, s) { return Math.max(max, s.wind); }, 0);

        document.getElementById('total-storms').textContent = totalStorms;
        document.getElementById('total-hurricanes').textContent = hurricanes;
        document.getElementById('total-major').textContent = major;
        document.getElementById('max-wind').textContent = maxWind || '--';
        document.getElementById('storm-count').textContent = totalStorms;
    }

    function showStormDetail(storm) {
        var panel = document.getElementById('storm-detail-panel');
        var content = document.getElementById('storm-detail-content');
        var cat = getCategory(storm.wind);

        panel.style.display = 'block';
        content.innerHTML =
            '<div class="detail-row"><span class="detail-label">Name</span><span class="detail-value">' + storm.name + '</span></div>' +
            '<div class="detail-row"><span class="detail-label">ID</span><span class="detail-value">' + storm.id + '</span></div>' +
            '<div class="detail-row"><span class="detail-label">Category</span><span class="detail-value"><span class="storm-category ' + cat.cls + '">' + cat.label + '</span></span></div>' +
            '<div class="detail-row"><span class="detail-label">Position</span><span class="detail-value">' + storm.lat.toFixed(1) + '&deg;N, ' + Math.abs(storm.lon).toFixed(1) + '&deg;' + (storm.lon < 0 ? 'W' : 'E') + '</span></div>' +
            '<div class="detail-row"><span class="detail-label">Max Wind</span><span class="detail-value">' + storm.wind + ' kt (' + Math.round(storm.wind * 1.151) + ' mph)</span></div>' +
            '<div class="detail-row"><span class="detail-label">Pressure</span><span class="detail-value">' + storm.pressure + ' mb</span></div>' +
            '<div class="detail-row"><span class="detail-label">Movement</span><span class="detail-value">' + storm.movement + '</span></div>' +
            '<div class="detail-row"><span class="detail-label">Advisory</span><span class="detail-value">' + storm.advisory + '</span></div>';

        if (storm.forecast && storm.forecast.length > 0) {
            content.innerHTML += '<h3 style="margin-top:12px; font-size:0.85rem; color:var(--text-secondary);">FORECAST</h3>';
            storm.forecast.forEach(function (pt) {
                var fcCat = getCategory(pt.wind);
                content.innerHTML +=
                    '<div class="detail-row"><span class="detail-label">' + pt.time + '</span><span class="detail-value">' + pt.wind + ' kt <span class="storm-category ' + fcCat.cls + '" style="font-size:0.65rem;">' + fcCat.label + '</span></span></div>';
            });
        }
    }

    // --- Basin Filtering ---
    function filterByBasin(storms, basin) {
        if (basin === 'all') return storms;

        var basinPrefixes = {
            'NA': ['AL'],
            'EP': ['EP'],
            'WP': ['WP'],
            'NI': ['IO', 'NI'],
            'SI': ['SH', 'SI'],
            'SP': ['SP'],
        };

        var prefixes = basinPrefixes[basin] || [];
        return storms.filter(function (s) {
            return prefixes.some(function (p) {
                return s.id.startsWith(p);
            }) || s.basin === basin;
        });
    }

    // --- Main ---
    async function loadDashboard() {
        allStorms = await fetchActiveStorms();

        var basin = document.getElementById('basin-filter').value;
        var filtered = filterByBasin(allStorms, basin);

        renderStorms(filtered);
        updateStormList(filtered);
        updateStats(filtered);
    }

    function init() {
        initMap();
        loadDashboard();

        document.getElementById('basin-filter').addEventListener('change', function () {
            var basin = this.value;
            var filtered = filterByBasin(allStorms, basin);
            renderStorms(filtered);
            updateStormList(filtered);
            updateStats(filtered);
        });

        document.getElementById('refresh-btn').addEventListener('click', function () {
            loadDashboard();
        });

        // Auto-refresh every 5 minutes
        setInterval(loadDashboard, 5 * 60 * 1000);
    }

    // Boot
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
