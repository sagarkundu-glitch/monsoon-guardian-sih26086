/* Monsoon Guardian — final app.js for the current index.html/style.css */
(() => {
  "use strict";
  if (window.__MG_STARTED__) return;
  window.__MG_STARTED__ = true;

  // The dashboard and API are served from the same host in deployment.
  const API = window.location.origin;
  const BOUNDS = {
    south: 21.5,
    north: 27.2,
    west: 85.8,
    east: 89.9
  };

  const DEFAULTS = {
    lat: 22.50,
    lon: 88.00,
    date: "2024-10-01"
  };

  const MIN_DATE = "2024-01-30";
  const MAX_DATE = "2024-12-01";

  const METRICS = {
    onset7Accuracy: 0.715295,
    onset14Auc: 0.875278,
    onset30Auc: 0.913578,
    break7Auc: 0.816895,
    break14Auc: 0.847366,
    break30Auc: 0.816388
  };

  const ENSO = {
    1: 0.70,
    2: 0.68,
    3: 0.78,
    4: 0.34,
    5: 0.12,
    6: -0.24,
    7: -0.72,
    8: -0.73,
    9: -0.65,
    10: -0.51,
    11: -0.68,
    12: -0.91
  };

  const IOD = {
    1: 0.765,
    2: 0.327,
    3: 0.421,
    4: 0.440,
    5: 0.298,
    6: 0.197,
    7: 0.033,
    8: 0.267,
    9: 0.115,
    10: -0.196,
    11: -0.382,
    12: -0.331
  };


  const state = {
    map: null,
    gridLayer: null,
    riskLayer: null,
    rainfallLayer: null,
    marker: null,

    grid: [],
    latest: null,
    raw: null,

    selectedHorizon: 7,
    activeLayer: "grid",

    language: "en",
    crop: "rice",

    predictionRunning: false,
    backendHealthy: false,
    liveWeatherTimer: null,
    liveWeatherRequestId: 0
  };


  /* ============================================================
     DOM HELPERS
  ============================================================ */

  const $ = id =>
    document.getElementById(id);

  const $$ = selector =>
    Array.from(
      document.querySelectorAll(selector)
    );


  const text = (
    id,
    value
  ) => {
    const element = $(id);

    if (element) {
      element.textContent =
        value == null
          ? ""
          : String(value);
    }

    return element;
  };


  const html = (
    id,
    value
  ) => {
    const element = $(id);

    if (element) {
      element.innerHTML =
        value == null
          ? ""
          : String(value);
    }

    return element;
  };


  const show = id => {
    const element = $(id);

    if (element) {
      element.hidden = false;
      element.classList.remove(
        "hidden"
      );
    }

    return element;
  };


  const hide = id => {
    const element = $(id);

    if (element) {
      element.hidden = true;
      element.classList.add(
        "hidden"
      );
    }

    return element;
  };


  const esc = value =>
    String(value ?? "")
      .replaceAll(
        "&",
        "&amp;"
      )
      .replaceAll(
        "<",
        "&lt;"
      )
      .replaceAll(
        ">",
        "&gt;"
      )
      .replaceAll(
        '"',
        "&quot;"
      )
      .replaceAll(
        "'",
        "&#039;"
      );


  const num = value => {
    const number = Number(value);

    return Number.isFinite(number)
      ? number
      : NaN;
  };


  const clamp = (
    value,
    minimum,
    maximum
  ) =>
    Math.min(
      maximum,
      Math.max(
        minimum,
        num(value) || 0
      )
    );


  const pct = value => {
    const number = num(value);

    if (
      !Number.isFinite(
        number
      )
    ) {
      return NaN;
    }

    if (
      number >= 0 &&
      number <= 1
    ) {
      return number * 100;
    }

    return clamp(
      number,
      0,
      100
    );
  };


  const fpct = (
    value,
    decimals = 1
  ) =>
    Number.isFinite(
      pct(value)
    )
      ? `${pct(value).toFixed(
          decimals
        )}%`
      : "—";


  const fnum = (
    value,
    decimals = 2
  ) =>
    Number.isFinite(
      num(value)
    )
      ? num(value).toFixed(
          decimals
        )
      : "—";


  const dateNorm = value =>
    String(
      value || ""
    ).slice(
      0,
      10
    );


  const fdate = value => {
    const normalized =
      dateNorm(value);

    if (
      !normalized
    ) {
      return "—";
    }

    const date =
      new Date(
        `${normalized}T00:00:00`
      );

    if (
      Number.isNaN(
        date.getTime()
      )
    ) {
      return normalized;
    }

    return date.toLocaleDateString(
      "en-IN",
      {
        day: "2-digit",
        month: "short",
        year: "numeric"
      }
    );
  };


  const month = value => {
    const date =
      new Date(
        `${dateNorm(
          value
        ) || DEFAULTS.date}T00:00:00`
      );

    if (
      Number.isNaN(
        date.getTime()
      )
    ) {
      return 10;
    }

    return (
      date.getMonth() + 1
    );
  };


  const inside = (
    latitude,
    longitude
  ) => {

    const lat =
      num(latitude);

    const lon =
      num(longitude);

    return (
      Number.isFinite(lat) &&
      Number.isFinite(lon) &&
      lat >= BOUNDS.south &&
      lat <= BOUNDS.north &&
      lon >= BOUNDS.west &&
      lon <= BOUNDS.east
    );
  };


  /* ============================================================
     ERROR / LOADING
  ============================================================ */

  function error(message) {

    console.error(
      "[Monsoon Guardian]",
      message
    );

    const element =
      $("error");

    if (element) {
      element.textContent =
        String(message);

      element.hidden = false;

      element.classList.remove(
        "hidden"
      );
    }
  }


  function clearError() {

    const element =
      $("error");

    if (element) {
      element.textContent = "";

      element.hidden = true;

      element.classList.add(
        "hidden"
      );
    }
  }


  function analysisState(
    message
  ) {

    text(
      "analysisStateText",
      message
    );
  }


  function toggleLoading(
    id,
    active
  ) {

    const element =
      $(id);

    if (!element) {
      return;
    }

    element.hidden =
      !active;

    element.classList.toggle(
      "hidden",
      !active
    );
  }


  function loading(active) {

    state.predictionRunning =
      Boolean(active);

    toggleLoading(
      "loading",
      active
    );

    const button =
      $("predictButton");

    if (!button) {
      return;
    }

    if (active) {

      button.disabled = true;

      button.dataset.oldText =
        button.dataset.oldText ||
        button.textContent;

      button.textContent =
        "Analyzing Monsoon...";

    } else {

      button.disabled = false;

      button.textContent =
        button.dataset.oldText ||
        "Analyze Monsoon";
    }
  }


  /* ============================================================
     API
  ============================================================ */

  async function request(
    path,
    options = {},
    timeout = 45000
  ) {

    const controller =
      new AbortController();

    const timer =
      setTimeout(
        () =>
          controller.abort(),
        timeout
      );

    try {

      const response =
        await fetch(
          `${API}${path}`,
          {
            ...options,
            signal:
              controller.signal
          }
        );

      const raw =
        await response.text();

      let data = null;

      try {

        data =
          raw
            ? JSON.parse(raw)
            : null;

      } catch {

        data = {
          detail: raw
        };
      }

      if (
        !response.ok
      ) {

        throw new Error(
          data?.detail ||
          data?.message ||
          `Request failed (${response.status})`
        );
      }

      return data;

    } catch (
      fetchError
    ) {

      if (
        fetchError?.name ===
        "AbortError"
      ) {

        throw new Error(
          "Request timed out. Please make sure the FastAPI backend is running."
        );
      }

      throw fetchError;

    } finally {

      clearTimeout(
        timer
      );
    }
  }


  async function health() {

    try {

      const data =
        await request(
          "/health",
          {
            headers: {
              Accept:
                "application/json"
            }
          },
          6000
        );

      state.backendHealthy =
        data?.status === "ok" ||
        data?.status === "healthy" ||
        data?.engine === "ready" ||
        data?.engine_ready === true;

      text(
        "engineStatusText",
        state.backendHealthy
          ? "Prediction engine online"
          : "Prediction engine not ready"
      );

      return state.backendHealthy;

    } catch (
      healthError
    ) {

      state.backendHealthy =
        false;

      text(
        "engineStatusText",
        "Prediction engine offline"
      );

      console.warn(
        "Backend health check failed:",
        healthError
      );

      return false;
    }
  }


  async function info() {

    try {

      const data =
        await request(
          "/info",
          {
            headers: {
              Accept:
                "application/json"
            }
          },
          6000
        );

      state.info =
        data;

      return data;

    } catch (
      infoError
    ) {

      console.info(
        "/info unavailable",
        infoError
      );

      return null;
    }
  }


  /* ============================================================
     INPUTS
  ============================================================ */

  function setupInputs() {

    const latitude =
      $("latitude");

    const longitude =
      $("longitude");

    const date =
      $("date");


    if (
      latitude &&
      !latitude.value
    ) {
      latitude.value =
        DEFAULTS.lat;
    }


    if (
      longitude &&
      !longitude.value
    ) {
      longitude.value =
        DEFAULTS.lon;
    }


    if (date) {

      date.min =
        MIN_DATE;

      date.max =
        MAX_DATE;

      if (!date.value) {
        date.value =
          DEFAULTS.date;
      }

      date.addEventListener(
        "change",
        () => {

          text(
            "analysisDate",
            fdate(
              date.value
            )
          );
        }
      );
    }


    const sync =
      () => {

        const lat =
          num(
            latitude?.value
          );

        const lon =
          num(
            longitude?.value
          );

        if (
          inside(
            lat,
            lon
          )
        ) {

          setLocation(
            lat,
            lon,
            false
          );
        }
      };


    latitude?.addEventListener(
      "change",
      sync
    );


    longitude?.addEventListener(
      "change",
      sync
    );


    text(
      "analysisDate",
      fdate(
        date?.value ||
        DEFAULTS.date
      )
    );
  }


  function inputs() {

    return {

      lat:
        num(
          $("latitude")
            ?.value
        ),

      lon:
        num(
          $("longitude")
            ?.value
        ),

      date:
        dateNorm(
          $("date")
            ?.value
        )
    };
  }


  function validate(
    values
  ) {

    if (
      !Number.isFinite(
        values.lat
      )
    ) {
      return (
        "Please enter a valid latitude."
      );
    }


    if (
      !Number.isFinite(
        values.lon
      )
    ) {
      return (
        "Please enter a valid longitude."
      );
    }


    if (
      values.lat < -90 ||
      values.lat > 90
    ) {
      return (
        "Latitude must be between -90 and 90."
      );
    }


    if (
      values.lon < -180 ||
      values.lon > 180
    ) {
      return (
        "Longitude must be between -180 and 180."
      );
    }


    if (
      !values.date
    ) {
      return (
        "Please select a prediction date."
      );
    }


    if (
      values.date <
        MIN_DATE ||
      values.date >
        MAX_DATE
    ) {
      return (
        `Prediction date must be between ${MIN_DATE} and ${MAX_DATE}.`
      );
    }


    if (
      !inside(
        values.lat,
        values.lon
      )
    ) {
      return (
        `Select a location inside the prototype study region: ` +
        `${BOUNDS.south}–${BOUNDS.north}°N, ` +
        `${BOUNDS.west}–${BOUNDS.east}°E.`
      );
    }


    return "";
  }


  function setLocation(
    latitude,
    longitude,
    move = true
  ) {

    const lat =
      num(latitude);

    const lon =
      num(longitude);


    if (
      $("latitude")
    ) {
      $("latitude").value =
        lat.toFixed(2);
    }


    if (
      $("longitude")
    ) {
      $("longitude").value =
        lon.toFixed(2);
    }


    text(
      "gridLocation",
      `${lat.toFixed(2)}°N, ${lon.toFixed(2)}°E`
    );


    text(
      "analysisDate",
      fdate(
        $("date")
          ?.value ||
        DEFAULTS.date
      )
    );


    if (move) {

      setMarker(
        lat,
        lon,
        "Selected Location"
      );
    }

    clearTimeout(state.liveWeatherTimer);
    state.liveWeatherTimer = setTimeout(
      () => loadLiveWeather(lat, lon),
      350
    );
  }


  async function loadLiveWeather(latitude, longitude) {

    const lat = num(latitude);
    const lon = num(longitude);
    const requestId = ++state.liveWeatherRequestId;
    const button = $("liveWeatherRefresh");
    const body = $("liveWeatherBody");

    if (!Number.isFinite(lat) || !Number.isFinite(lon)) {
      return;
    }

    text("liveWeatherStatus", "Loading current conditions…");
    text("liveWeatherMessage", "");
    text("liveWeatherTemperature", "—");
    text("liveWeatherHumidity", "—");
    text("liveWeatherRain", "—");
    text("liveWeatherWind", "—");
    text("liveWeatherToday", "Waiting for weather data");
    text("liveWeatherUpdated", `Requesting weather for ${lat.toFixed(2)}°, ${lon.toFixed(2)}°; location is rounded to 0.01°.`);
    if (body) body.setAttribute("aria-busy", "true");
    if (button) button.disabled = true;

    const fmt = (value, digits = 0) => {
      if (value == null || !Number.isFinite(Number(value))) return "—";
      return Number(value).toFixed(digits);
    };

    try {
      const params = new URLSearchParams({
        latitude: lat.toFixed(2),
        longitude: lon.toFixed(2)
      });
      const weather = await request(
        `/weather/live?${params.toString()}`,
        { headers: { Accept: "application/json" } },
        12000
      );
      if (requestId !== state.liveWeatherRequestId) return;
      const current = weather?.current || {};
      const today = weather?.today || {};

      text("liveWeatherTemperature", `${fmt(current.temperature_c, 1)} °C`);
      text("liveWeatherHumidity", `${fmt(current.relative_humidity_pct)}%`);
      text("liveWeatherRain", `${fmt(current.rain_mm, 1)} mm`);
      text("liveWeatherWind", `${fmt(current.wind_speed_kmh, 1)} km/h`);

      const high = fmt(today.temperature_max_c, 0);
      const low = fmt(today.temperature_min_c, 0);
      const chance = fmt(today.precipitation_probability_max_pct, 0);
      const total = fmt(today.precipitation_sum_mm, 1);
      text(
        "liveWeatherToday",
        `High ${high}°C · low ${low}°C · rain chance ${chance}% · forecast rain ${total} mm`
      );

      const observedAt = weather?.observed_at
        ? ` · observation ${weather.observed_at} (${weather.timezone || "local time"})`
        : "";
      text("liveWeatherStatus", "Weather data available");
      text(
        "liveWeatherUpdated",
        `Weather comes from Open-Meteo and is separate from the 2024-trained model; location rounded to 0.01°${observedAt}.`
      );
    } catch (weatherError) {
      if (requestId !== state.liveWeatherRequestId) return;
      text("liveWeatherStatus", "Weather unavailable");
      text(
        "liveWeatherToday",
        "Live weather could not be loaded for this location."
      );
      text(
        "liveWeatherMessage",
        weatherError?.message || "Please try refreshing the weather data."
      );
      console.warn("Live weather request failed:", weatherError);
    } finally {
      if (requestId === state.liveWeatherRequestId) {
        if (body) body.setAttribute("aria-busy", "false");
        if (button) button.disabled = false;
      }
    }
  }


  /* ============================================================
     MAP
  ============================================================ */

  function initMap() {

    const element =
      $("predictionMap");

    if (
      !element ||
      typeof L ===
        "undefined"
    ) {

      console.warn(
        "Leaflet or map element is unavailable."
      );

      return;
    }


    if (state.map) {

      setTimeout(
        () =>
          state.map?.invalidateSize(),
        100
      );

      return;
    }


    state.map =
      L.map(
        element,
        {
          center:
            [23.85, 87.80],

          zoom:
            6,

          minZoom:
            5,

          maxZoom:
            13,

          zoomControl:
            true
        }
      );


    L.tileLayer(
      "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
      {
        maxZoom:
          19,

        attribution:
          "&copy; OpenStreetMap contributors"
      }
    )
      .addTo(
        state.map
      );


    L.rectangle(
      [
        [
          BOUNDS.south,
          BOUNDS.west
        ],

        [
          BOUNDS.north,
          BOUNDS.east
        ]
      ],
      {
        weight:
          2,

        fillOpacity:
          0.03,

        interactive:
          false
      }
    )
      .addTo(
        state.map
      );


    state.map.fitBounds(
      [
        [
          BOUNDS.south,
          BOUNDS.west
        ],

        [
          BOUNDS.north,
          BOUNDS.east
        ]
      ],
      {
        padding:
          [18, 18]
      }
    );


    state.map.on(
      "click",
      event => {

        const latitude =
          Number(
            event.latlng.lat
              .toFixed(2)
          );

        const longitude =
          Number(
            event.latlng.lng
              .toFixed(2)
          );


        if (
          !inside(
            latitude,
            longitude
          )
        ) {

          error(
            "Please select a location inside the supported study region."
          );

          return;
        }


        clearError();


        setLocation(
          latitude,
          longitude
        );


        analysisState(
          "Analysis Ready"
        );
      }
    );


    setTimeout(
      () =>
        state.map?.invalidateSize(),
      300
    );
  }


  function setMarker(
    latitude,
    longitude,
    title =
      "Selected Location"
  ) {

    if (!state.map) {
      return;
    }


    if (state.marker) {

      try {

        state.map.removeLayer(
          state.marker
        );

      } catch {
        /* ignore */
      }

      state.marker =
        null;
    }


    state.marker =
      L.circleMarker(
        [
          latitude,
          longitude
        ],
        {
          radius:
            8,

          weight:
            3,

          fillOpacity:
            0.85
        }
      )
        .addTo(
          state.map
        )
        .bindPopup(
          `<strong>${esc(title)}</strong><br>` +
          `${fnum(latitude)}°N, ` +
          `${fnum(longitude)}°E`
        );


    try {

      state.map.setView(
        [
          latitude,
          longitude
        ],
        Math.max(
          state.map.getZoom(),
          7
        ),
        {
          animate:
            true
        }
      );

    } catch {
      /* ignore */
    }
  }


  /* ============================================================
     GRID
  ============================================================ */

  async function loadLocations() {

    try {

      toggleLoading(
        "mapLoading",
        true
      );


      const data =
        await request(
          "/locations",
          {
            headers: {
              Accept:
                "application/json"
            }
          },
          12000
        );


      const locations =
        Array.isArray(data)
          ? data
          : (
              data?.locations ||
              data?.data ||
              data?.points ||
              data?.grid ||
              data?.results ||
              []
            );


      state.grid =
        locations
          .map(
            item => ({
              lat:
                num(
                  item.latitude ??
                  item.lat ??
                  item.y
                ),

              lon:
                num(
                  item.longitude ??
                  item.lon ??
                  item.lng ??
                  item.x
                )
            })
          )
          .filter(
            point =>
              inside(
                point.lat,
                point.lon
              )
          );


      renderGrid();


      text(
        "gridStatusText",
        `${state.grid.length} grid cells`
      );


    } catch (
      locationsError
    ) {

      state.grid = [];

      text(
        "gridStatusText",
        "Grid data unavailable"
      );


      console.warn(
        "Grid locations unavailable:",
        locationsError
      );


    } finally {

      toggleLoading(
        "mapLoading",
        false
      );
    }
  }


  function renderGrid() {

    if (!state.map) {
      return;
    }


    if (
      state.gridLayer
    ) {

      try {

        state.map.removeLayer(
          state.gridLayer
        );

      } catch {
        /* ignore */
      }
    }


    if (
      !state.grid.length
    ) {
      return;
    }


    const group =
      L.layerGroup();


    state.grid.forEach(
      point => {

        const marker =
          L.circleMarker(
            [
              point.lat,
              point.lon
            ],
            {
              radius:
                3.5,

              weight:
                1,

              fillOpacity:
                0.65
            }
          );


        marker.bindTooltip(
          `${point.lat.toFixed(2)}°N, ` +
          `${point.lon.toFixed(2)}°E`
        );


        marker.on(
          "click",
          event => {

            L.DomEvent
              .stopPropagation(
                event
              );


            setLocation(
              point.lat,
              point.lon
            );


            analysisState(
              "Analysis Ready"
            );


            clearError();
          }
        );


        marker.addTo(
          group
        );
      }
    );


    group.addTo(
      state.map
    );


    state.gridLayer =
      group;
  }


  function clearMapExtra() {

    for (
      const layerName of
        [
          "riskLayer",
          "rainfallLayer"
        ]
    ) {

      const layer =
        state[layerName];

      if (
        layer &&
        state.map
      ) {

        try {

          state.map.removeLayer(
            layer
          );

        } catch {
          /* ignore */
        }
      }


      state[layerName] =
        null;
    }
  }


  function updateLayer() {

    clearMapExtra();


    if (!state.map) {
      return;
    }


    const layer =
      state.activeLayer;


    /* GRID */
    if (
      layer ===
      "grid"
    ) {

      text(
        "mapInstruction",
        "Grid layer active — click a grid point to select a location."
      );

      return;
    }


    /* RAINFALL */
    if (
      layer ===
      "rainfall"
    ) {

      const group =
        L.layerGroup();


      state.grid.forEach(
        point => {

          L.circle(
            [
              point.lat,
              point.lon
            ],
            {
              radius:
                10000,

              weight:
                1,

              fillOpacity:
                0.06
            }
          )
            .bindTooltip(
              "Rainfall intensity is not returned by the current API."
            )
            .addTo(
              group
            );
        }
      );


      group.addTo(
        state.map
      );


      state.rainfallLayer =
        group;


      text(
        "mapInstruction",
        "Rainfall layer — intensity data is not exposed by the current API."
      );


      return;
    }


    /* ONSET / BREAK */
    const key =
      String(
        state.selectedHorizon
      );


    const prediction =
      state.latest?.[
        layer
      ]?.[
        key
      ];


    if (!prediction) {

      text(
        "mapInstruction",
        `Run an analysis to view ${layer} probability.`
      );

      return;
    }


    const location =
      state.latest
        .gridLocation ||
      state.latest
        .requestedLocation;


    const probability =
      pct(
        prediction.probability
      );


    if (
      !inside(
        location?.latitude,
        location?.longitude
      ) ||
      !Number.isFinite(
        probability
      )
    ) {
      return;
    }


    const color =
      probability >= 70
        ? "#dc2626"
        : probability >= 45
          ? "#f59e0b"
          : "#2563eb";


    state.riskLayer =
      L.circle(
        [
          location.latitude,
          location.longitude
        ],
        {
          radius:
            30000,

          color:
            color,

          fillColor:
            color,

          weight:
            3,

          fillOpacity:
            0.16
        }
      )
        .addTo(
          state.map
        )
        .bindPopup(
          `<strong>${
            layer ===
            "onset"
              ? "Monsoon onset"
              : "Monsoon break"
          }</strong><br>` +
          `${state.selectedHorizon}-day probability: ` +
          `${probability.toFixed(1)}%`
        );


    text(
      "mapInstruction",
      `${
        layer ===
        "onset"
          ? "Onset"
          : "Break"
      } layer · ` +
      `${state.selectedHorizon}-day probability ` +
      `${probability.toFixed(1)}%`
    );
  }


  /* ============================================================
     PREDICTION NORMALIZATION
  ============================================================ */

  function normalizeGroup(
    group
  ) {

    const output = {};


    for (
      const days of
        [
          7,
          14,
          30
        ]
    ) {

      const raw =
        group?.[
          `${days}_day`
        ] ??
        group?.[
          days
        ] ??
        group?.[
          `${days}d`
        ] ??
        null;


      output[
        String(days)
      ] = {

        probability:
          raw?.probability ??
          raw?.prob ??
          raw?.percent ??
          raw?.risk,

        threshold:
          raw?.threshold ??
          null,

        signal:
          Boolean(
            raw?.prototype_signal ??
            raw?.prototypeSignal ??
            raw?.signal
          )
      };
    }


    return output;
  }


  function normalize(
    data
  ) {

    return {

      requestedLocation: {

        latitude:
          num(
            data
              ?.requested_location
              ?.latitude ??
            data
              ?.requestedLocation
              ?.latitude
          ),

        longitude:
          num(
            data
              ?.requested_location
              ?.longitude ??
            data
              ?.requestedLocation
              ?.longitude
          )
      },


      gridLocation: {

        latitude:
          num(
            data
              ?.grid_location
              ?.latitude ??
            data
              ?.gridLocation
              ?.latitude ??
            data
              ?.requested_location
              ?.latitude
          ),

        longitude:
          num(
            data
              ?.grid_location
              ?.longitude ??
            data
              ?.gridLocation
              ?.longitude ??
            data
              ?.requested_location
              ?.longitude
          )
      },


      date:
        dateNorm(
          data?.date
        ),


      onset:
        normalizeGroup(
          data?.onset
        ),


      break:
        normalizeGroup(
          data?.break
        ),


      rainfall:
        data?.rainfall ??
        data?.rainfall_context ??
        null,


      climate:
        data?.climate ??
        data?.climate_context ??
        null,


      heavy:
        data?.heavy_rainfall ??
        data?.heavyRainfall ??
        null,


      info:
        data?.model_information ??
        data?.modelInformation ??
        null
    };
  }


  /* ============================================================
     MAIN PREDICTION
  ============================================================ */

  async function runPrediction(
    event
  ) {

    event?.preventDefault();


    if (
      state.predictionRunning
    ) {
      return;
    }


    clearError();


    const values =
      inputs();


    const validationError =
      validate(
        values
      );


    if (
      validationError
    ) {

      error(
        validationError
      );

      return;
    }


    analysisState(
      "Running prediction…"
    );


    loading(
      true
    );


    try {

      const healthy =
        await health();


      if (!healthy) {

        throw new Error(
          "Prediction engine is not ready. Start the FastAPI backend on port 8000."
        );
      }


      const data =
        await request(
          "/predict",
          {
            method:
              "POST",

            headers: {
              "Content-Type":
                "application/json",

              Accept:
                "application/json"
            },

            body:
              JSON.stringify(
                {
                  latitude:
                    values.lat,

                  longitude:
                    values.lon,

                  date:
                    values.date
                }
              )
          }
        );


      state.raw =
        data;


      state.latest =
        normalize(
          data
        );


      renderAll();


      setMarker(
        state.latest
          .gridLocation
          .latitude,

        state.latest
          .gridLocation
          .longitude,

        "Analyzed Grid Location"
      );


      analysisState(
        "Analysis complete"
      );


      show(
        "results"
      );


      setTimeout(
        () => {

          $(
            "results"
          )?.scrollIntoView(
            {
              behavior:
                "smooth",

              block:
                "start"
            }
          );

        },
        100
      );


      return data;


    } catch (
      predictionError
    ) {

      analysisState(
        "Prediction failed"
      );


      error(
        predictionError?.message ||
        "Prediction failed."
      );


      return null;


    } finally {

      loading(
        false
      );
    }
  }


  /*
   * Global function for browser console / integrations.
   */
  window.runPrediction =
    runPrediction;


  /* ============================================================
     RENDER ALL RESULTS
  ============================================================ */

  function renderAll() {

    const prediction =
      state.latest;


    const location =
      prediction
        .gridLocation ||
      prediction
        .requestedLocation;


    text(
      "gridLocation",
      `${fnum(location.latitude)}°N, ` +
      `${fnum(location.longitude)}°E`
    );


    text(
      "analysisDate",
      fdate(
        prediction.date
      )
    );


    text(
      "currentLocationText",
      `${fnum(location.latitude)}°N, ` +
      `${fnum(location.longitude)}°E`
    );


    text(
      "currentDateText",
      fdate(
        prediction.date
      )
    );


    text(
      "modelTypeText",
      "Random Forest prototype"
    );


    text(
      "modelStatusText",
      prediction.info
        ?.note ||
      prediction.info
        ?.prediction_type ||
      "Experimental model"
    );


    /* ========================================================
       7-DAY OVERVIEW
    ======================================================== */

    const onset7 =
      pct(
        prediction
          .onset
          ["7"]
          ?.probability
      );


    const break7 =
      pct(
        prediction
          .break
          ["7"]
          ?.probability
      );


    text(
      "overviewOnset",
      fpct(onset7)
    );


    text(
      "overviewBreak",
      fpct(break7)
    );


    setWidth(
      "overviewOnsetBar",
      onset7
    );


    setWidth(
      "overviewBreakBar",
      break7
    );


    text(
      "overviewOnsetSignal",
      riskLabel(
        onset7
      )
    );


    text(
      "overviewBreakSignal",
      riskLabel(
        break7
      )
    );


    const outlook =
      overall(
        onset7,
        break7
      );


    text(
      "overviewStatus",
      outlook.title
    );


    text(
      "overviewStatusCaption",
      outlook.caption
    );


    text(
      "overviewStatusText",
      outlook.short
    );


    const statusDot =
      $("overviewStatusDot");


    if (statusDot) {

      statusDot.className =
        "status-dot";

      statusDot.classList.add(
        outlook.status
      );
    }


    /* ========================================================
       SNAPSHOT
    ======================================================== */

    for (
      const days of
        [
          7,
          14,
          30
        ]
    ) {

      text(
        `snapshotOnset${days}`,
        fpct(
          prediction
            .onset[
              String(days)
            ]
            ?.probability
        )
      );


      text(
        `snapshotBreak${days}`,
        fpct(
          prediction
            .break[
              String(days)
            ]
            ?.probability
        )
      );
    }


    /* ========================================================
       CARDS
    ======================================================== */

    renderCards(
      "onsetCards",
      prediction.onset,
      "onset"
    );


    renderCards(
      "breakCards",
      prediction.break,
      "break"
    );


    /* ========================================================
       OTHER MODULES
    ======================================================== */

    renderRainfall(
      prediction.rainfall
    );


    renderClimate(
      prediction.climate,
      prediction.date
    );


    renderHeavy(
      prediction.heavy
    );


    renderAdvisory();


    renderDrivers();


    renderMetrics();


    updateLayer();
  }


  function setWidth(
    id,
    value
  ) {

    const element =
      $(id);

    if (!element) {
      return;
    }


    const percentage =
      Number.isFinite(
        num(value)
      )
        ? clamp(
            value,
            0,
            100
          )
        : 0;


    element.style.width =
      `${percentage}%`;
  }


  function riskLabel(
    value
  ) {

    if (
      !Number.isFinite(
        num(value)
      )
    ) {
      return "Unavailable";
    }


    if (
      value >= 70
    ) {
      return "High";
    }


    if (
      value >= 45
    ) {
      return "Moderate";
    }


    return "Low";
  }


  function overall(
    onset,
    breakProbability
  ) {

    if (
      !Number.isFinite(onset) ||
      !Number.isFinite(
        breakProbability
      )
    ) {

      return {

        title:
          "Awaiting",

        caption:
          "Run analysis to calculate the experimental outlook.",

        short:
          "Run analysis",

        status:
          "neutral"
      };
    }


    if (
      onset >= 70 &&
      breakProbability >= 70
    ) {

      return {

        title:
          "Mixed signal",

        caption:
          "Both experimental probabilities are elevated in the 7-day outlook.",

        short:
          "Mixed rainfall signals",

        status:
          "mixed"
      };
    }


    if (
      breakProbability >= 70
    ) {

      return {

        title:
          "Break signal elevated",

        caption:
          "The experimental 7-day break probability is elevated.",

        short:
          "Break signal",

        status:
          "break"
      };
    }


    if (
      onset >= 70
    ) {

      return {

        title:
          "Onset signal elevated",

        caption:
          "The experimental 7-day onset probability is elevated.",

        short:
          "Onset signal",

        status:
          "onset"
      };
    }


    if (
      onset >= 45 ||
      breakProbability >= 45
    ) {

      return {

        title:
          "Moderate signal",

        caption:
          "At least one experimental 7-day probability is in the moderate range.",

        short:
          "Monitor",

        status:
          "neutral"
      };
    }


    return {

      title:
        "Lower signal",

      caption:
        "Both experimental 7-day probabilities are below the moderate range.",

      short:
        "Lower signal",

      status:
        "neutral"
    };
  }


  function renderCards(
    id,
    group,
    type
  ) {

    const element =
      $(id);

    if (!element) {
      return;
    }


    element.innerHTML =
      [
        7,
        14,
        30
      ]
        .map(
          days => {

            const prediction =
              group[
                String(days)
              ];


            const probability =
              pct(
                prediction
                  ?.probability
              );


            return `
              <article class="prediction-card">

                <div class="prediction-card-header">

                  <div>

                    <span class="prediction-card-kicker">
                      ${
                        type ===
                        "onset"
                          ? "Monsoon Onset"
                          : "Monsoon Break"
                      }
                    </span>

                    <strong>
                      ${days} Days
                    </strong>

                  </div>


                  <span class="prediction-card-badge">
                    ${esc(
                      riskLabel(
                        probability
                      )
                    )}
                  </span>

                </div>


                <div class="prediction-card-probability">
                  ${
                    Number.isFinite(
                      probability
                    )
                      ? `${probability.toFixed(1)}%`
                      : "—"
                  }
                </div>


                <div class="prediction-progress">

                  <span
                    style="width:${
                      Number.isFinite(
                        probability
                      )
                        ? probability
                        : 0
                    }%"
                  ></span>

                </div>


                <p class="prediction-card-description">

                  Experimental rainfall-derived
                  ${type}
                  target probability.

                </p>


                <div class="prediction-card-meta">

                  <span>
                    Signal:
                    ${
                      prediction
                        ?.signal
                        ? "Positive"
                        : "Not positive"
                    }
                  </span>


                  <span>

                    ${
                      prediction
                        ?.threshold !==
                        null &&
                      prediction
                        ?.threshold !==
                        undefined
                        ? `Threshold ${fpct(
                            prediction.threshold,
                            0
                          )}`
                        : "Threshold unavailable"
                    }

                  </span>

                </div>

              </article>
            `;
          }
        )
        .join("");
  }


  /* ============================================================
     RAINFALL
  ============================================================ */

  function renderRainfall(
    rainfall
  ) {

    const values = [

      rainfall
        ?.total_7d ??
      rainfall
        ?.rainfall_7d ??
      rainfall
        ?.["7_day"],

      rainfall
        ?.total_14d ??
      rainfall
        ?.rainfall_14d ??
      rainfall
        ?.["14_day"],

      rainfall
        ?.total_30d ??
      rainfall
        ?.rainfall_30d ??
      rainfall
        ?.["30_day"]

    ];


    text(
      "rainfall7Value",
      Number.isFinite(
        num(values[0])
      )
        ? `${num(values[0]).toFixed(1)} mm`
        : "—"
    );


    text(
      "rainfall14Value",
      Number.isFinite(
        num(values[1])
      )
        ? `${num(values[1]).toFixed(1)} mm`
        : "—"
    );


    text(
      "rainfall30Value",
      Number.isFinite(
        num(values[2])
      )
        ? `${num(values[2]).toFixed(1)} mm`
        : "—"
    );


    text(
      "rainyDaysValue",
      rainfall
        ?.rainy_days ??
      rainfall
        ?.rainyDays ??
      "—"
    );


    text(
      "dryDaysValue",
      rainfall
        ?.dry_days ??
      rainfall
        ?.dryDays ??
      "—"
    );


    text(
      "drySpellValue",
      rainfall
        ?.consecutive_dry_days ??
      rainfall
        ?.max_dry_spell ??
      "—"
    );


    const available =
      values.some(
        value =>
          Number.isFinite(
            num(value)
          )
      );


    text(
      "rainfallChartStatus",
      available
        ? "Rainfall values returned by API"
        : "Rainfall values are not exposed by the current API"
    );


    if (available) {

      hide(
        "rainfallChartEmpty"
      );

      drawChart(
        values
      );

    } else {

      show(
        "rainfallChartEmpty"
      );

      drawChart(
        null
      );
    }
  }


  function drawChart(
    values
  ) {

    const canvas =
      $("rainfallChart");

    if (!canvas) {
      return;
    }


    const context =
      canvas.getContext(
        "2d"
      );

    if (!context) {
      return;
    }


    const rect =
      canvas.getBoundingClientRect();


    const width =
      Math.max(
        320,
        Math.floor(
          rect.width ||
          640
        )
      );


    const height =
      Math.max(
        210,
        Math.floor(
          rect.height ||
          240
        )
      );


    const ratio =
      window.devicePixelRatio ||
      1;


    canvas.width =
      width * ratio;

    canvas.height =
      height * ratio;


    context.setTransform(
      ratio,
      0,
      0,
      ratio,
      0,
      0
    );


    context.clearRect(
      0,
      0,
      width,
      height
    );


    if (!values) {
      return;
    }


    const numeric =
      values.map(
        value =>
          Number.isFinite(
            num(value)
          )
            ? num(value)
            : 0
      );


    const maximum =
      Math.max(
        ...numeric,
        1
      );


    const left =
      50;

    const top =
      20;

    const right =
      18;

    const bottom =
      40;


    const plotWidth =
      width -
      left -
      right;


    const plotHeight =
      height -
      top -
      bottom;


    context.strokeStyle =
      "#d8e2ed";

    context.fillStyle =
      "#536579";

    context.font =
      "12px Arial";


    for (
      let i = 0;
      i < 5;
      i++
    ) {

      const y =
        top +
        plotHeight -
        (
          plotHeight *
          i /
          4
        );


      context.beginPath();

      context.moveTo(
        left,
        y
      );

      context.lineTo(
        width -
        right,
        y
      );

      context.stroke();


      context.fillText(
        `${(
          maximum *
          i /
          4
        ).toFixed(0)} mm`,
        5,
        y + 4
      );
    }


    const barWidth =
      Math.min(
        72,
        plotWidth /
        6
      );


    numeric.forEach(
      (
        value,
        index
      ) => {

        const slot =
          plotWidth /
          3;


        const x =
          left +
          slot *
          index +
          (
            slot -
            barWidth
          ) /
          2;


        const barHeight =
          value /
          maximum *
          plotHeight;


        const y =
          top +
          plotHeight -
          barHeight;


        context.fillStyle =
          "#2563eb";


        context.fillRect(
          x,
          y,
          barWidth,
          barHeight
        );


        context.fillStyle =
          "#24364b";


        context.textAlign =
          "center";


        context.fillText(
          `${value.toFixed(1)} mm`,
          x +
            barWidth /
            2,
          Math.max(
            14,
            y - 7
          )
        );


        context.fillText(
          [
            "7d",
            "14d",
            "30d"
          ][index],
          x +
            barWidth /
            2,
          height - 13
        );
      }
    );


    context.textAlign =
      "start";
  }


  /* ============================================================
     CLIMATE
  ============================================================ */

  function renderClimate(
    climate,
    date
  ) {

    const selectedMonth =
      month(
        date
      );


    const backendENSO =
      num(
        climate?.enso ??
        climate?.ENSO ??
        climate?.mei ??
        climate?.MEI
      );


    const backendIOD =
      num(
        climate?.iod ??
        climate?.IOD ??
        climate?.dmi ??
        climate?.DMI
      );


    const enso =
      Number.isFinite(
        backendENSO
      )
        ? backendENSO
        : ENSO[
            selectedMonth
          ];


    const iod =
      Number.isFinite(
        backendIOD
      )
        ? backendIOD
        : IOD[
            selectedMonth
          ];


    text(
      "ensoValue",
      Number.isFinite(
        enso
      )
        ? enso.toFixed(2)
        : "—"
    );


    text(
      "iodValue",
      Number.isFinite(
        iod
      )
        ? iod.toFixed(3)
        : "—"
    );


    text(
      "ensoStatus",
      Number.isFinite(
        backendENSO
      )
        ? "API climate value"
        : "2024 MEI project context"
    );


    text(
      "iodStatus",
      Number.isFinite(
        backendIOD
      )
        ? "API climate value"
        : "2024 DMI project context"
    );


    const mjo =
      climate?.mjo ??
      climate?.MJO ??
      null;


    const phase =
      mjo?.phase ??
      climate?.mjo_phase;


    const amplitude =
      mjo?.amplitude ??
      climate?.mjo_amplitude;


    if (
      phase !== null &&
      phase !== undefined
    ) {

      text(
        "mjoValue",
        `Phase ${phase}${
          amplitude !==
          null &&
          amplitude !==
          undefined
            ? ` · Amp ${num(
                amplitude
              ).toFixed(2)}`
            : ""
        }`
      );


      text(
        "mjoStatus",
        "API MJO context"
      );

    } else {

      text(
        "mjoValue",
        "—"
      );


      text(
        "mjoStatus",
        "MJO phase not returned by current API"
      );
    }
  }


  /* ============================================================
     HEAVY RAIN
  ============================================================ */

  function renderHeavy(
    heavy
  ) {

    const probability =
      num(
        heavy?.probability ??
        heavy?.risk_probability ??
        heavy?.percent
      );


    text(
      "heavyRainfallStatus",
      Number.isFinite(
        probability
      )
        ? fpct(
            probability
          )
        : "EXTENSION MODULE"
    );


    text(
      "heavyRainfallModuleStatus",
      Number.isFinite(
        probability
      )
        ? "API-provided heavy-rainfall probability"
        : "Live heavy-rainfall probability is not returned by the current API."
    );
  }


  /* ============================================================
     ADVISORY
  ============================================================ */

  const ADVICE = {

    en: {

      onset: [
        "Onset preparation",
        "WATCH",
        "The experimental onset probability is elevated. Confirm sustained local rainfall and soil moisture before major sowing decisions."
      ],

      break: [
        "Break-phase caution",
        "CAUTION",
        "The experimental break probability is elevated. Prepare for possible dry-spell conditions and review irrigation options."
      ],

      mixed: [
        "Mixed monsoon signal",
        "MONITOR",
        "Onset and break probabilities both show meaningful signals. Use a flexible field plan and verify local conditions before high-cost decisions."
      ],

      neutral: [
        "Routine monitoring",
        "MONITOR",
        "The prototype does not show a strong 7-day onset or break signal. Continue local rainfall and soil-moisture monitoring."
      ]
    },


    bn: {

      onset: [
        "বর্ষা আগমনের প্রস্তুতি",
        "নজরদারি",
        "পরীক্ষামূলক মডেলে বর্ষা আগমনের সম্ভাবনা তুলনামূলক বেশি। বপনের আগে স্থানীয় বৃষ্টি ও মাটির আর্দ্রতা নিশ্চিত করুন।"
      ],

      break: [
        "বর্ষা বিরতি বিষয়ে সতর্কতা",
        "সতর্কতা",
        "পরীক্ষামূলক মডেলে শুষ্ক সময়ের সম্ভাবনা বেশি। মাটির আর্দ্রতা রক্ষা ও বিকল্প সেচের প্রস্তুতি রাখুন।"
      ],

      mixed: [
        "মিশ্র বর্ষা সংকেত",
        "পর্যবেক্ষণ",
        "বর্ষা আগমন ও বিরতি—দুই ধরনের সংকেতই রয়েছে। স্থানীয় পরিস্থিতি যাচাই করে নমনীয় কৃষি পরিকল্পনা রাখুন।"
      ],

      neutral: [
        "সাধারণ পর্যবেক্ষণ",
        "পর্যবেক্ষণ",
        "প্রোটোটাইপে ৭ দিনের বর্ষা আগমন বা বিরতির শক্তিশালী সংকেত নেই। স্থানীয় বৃষ্টি ও মাটির আর্দ্রতা পর্যবেক্ষণ চালিয়ে যান।"
      ]
    },


    hi: {

      onset: [
        "मानसून आगमन की तैयारी",
        "निगरानी",
        "प्रायोगिक मॉडल में मानसून आगमन की संभावना बढ़ी हुई है। बुवाई से पहले स्थानीय वर्षा और मिट्टी की नमी की पुष्टि करें।"
      ],

      break: [
        "मानसून ब्रेक में सावधानी",
        "सावधानी",
        "प्रायोगिक मॉडल में शुष्क अवधि की संभावना बढ़ी हुई है। मिट्टी की नमी संरक्षण और वैकल्पिक सिंचाई की तैयारी रखें।"
      ],

      mixed: [
        "मिश्रित मानसून संकेत",
        "निगरानी",
        "मानसून आगमन और ब्रेक दोनों के संकेत मौजूद हैं। स्थानीय परिस्थितियों की पुष्टि करके लचीली कृषि योजना रखें।"
      ],

      neutral: [
        "सामान्य निगरानी",
        "निगरानी",
        "प्रायोगिक मॉडल में 7 दिनों के लिए आगमन या ब्रेक का मजबूत संकेत नहीं है। स्थानीय मौसम और मिट्टी की नमी देखते रहें।"
      ]
    },


    or: {

      onset: [
        "ମୌସୁମୀ ବର୍ଷା ଆଗମନ ପାଇଁ ପ୍ରସ୍ତୁତି",
        "ନଜରଦାରି",
        "ପରୀକ୍ଷାମୂଳକ ମଡେଲରେ ମୌସୁମୀ ବର୍ଷା ଆଗମନର ସମ୍ଭାବନା ବଢ଼ିଛି। ବିଆ ବୋପିବା ପୂର୍ବରୁ ସ୍ଥାନୀୟ ବର୍ଷା ଓ ମାଟିର ଆର୍ଦ୍ରତା ଯାଞ୍ଚ କରନ୍ତୁ।"
      ],

      break: [
        "ମୌସୁମୀ ବିରତି ପାଇଁ ସାବଧାନତା",
        "ସାବଧାନ",
        "ଶୁଷ୍କ ସମୟର ସମ୍ଭାବନା ବଢ଼ିଛି। ମାଟିର ଆର୍ଦ୍ରତା ସଂରକ୍ଷଣ ଓ ବିକଳ୍ପ ଜଳସେଚନ ପାଇଁ ପ୍ରସ୍ତୁତ ରହନ୍ତୁ।"
      ],

      mixed: [
        "ମିଶ୍ରିତ ମୌସୁମୀ ସଙ୍କେତ",
        "ନଜରଦାରି",
        "ମୌସୁମୀ ବର୍ଷା ଆଗମନ ଓ ବିରତି ଉଭୟର ସଙ୍କେତ ରହିଛି। ସ୍ଥାନୀୟ ପରିସ୍ଥିତି ଯାଞ୍ଚ କରି ଲଚିଳା କୃଷି ଯୋଜନା ରଖନ୍ତୁ।"
      ],

      neutral: [
        "ସାଧାରଣ ନଜରଦାରି",
        "ନଜରଦାରି",
        "୭ ଦିନ ପାଇଁ ବର୍ଷା ଆଗମନ କିମ୍ବା ବିରତିର ଦୃଢ଼ ସଙ୍କେତ ନାହିଁ। ସ୍ଥାନୀୟ ପାଣିପାଗ ଓ ମାଟିର ଆର୍ଦ୍ରତା ନଜରରେ ରଖନ୍ତୁ।"
      ]
    }
  };


  const CROP = {
    rice:
      "Rice",

    maize:
      "Maize",

    pulses:
      "Pulses",

    groundnut:
      "Groundnut",

    cotton:
      "Cotton"
  };


  function adviceType() {

    const onset =
      pct(
        state.latest
          ?.onset
          ?.[
            "7"
          ]
          ?.probability
      );


    const breakProbability =
      pct(
        state.latest
          ?.break
          ?.[
            "7"
          ]
          ?.probability
      );


    if (
      onset >= 70 &&
      breakProbability < 40
    ) {
      return "onset";
    }


    if (
      breakProbability >= 70 &&
      onset < 40
    ) {
      return "break";
    }


    if (
      onset >= 40 ||
      breakProbability >= 40
    ) {
      return "mixed";
    }


    return "neutral";
  }


  function advisoryData() {

    const pack =
      ADVICE[
        state.language
      ] ||
      ADVICE.en;


    const advice =
      pack[
        adviceType()
      ] ||
      pack.neutral;


    const crop =
      CROP[
        state.crop
      ] ||
      "Crop";


    const actions = {

      en: [
        `Prepare ${crop.toLowerCase()} field operations according to actual local rainfall.`,
        "Check soil moisture before sowing or irrigation decisions.",
        "Review the forecast again before major field decisions."
      ],

      bn: [
        "স্থানীয় বৃষ্টি ও মাটির আর্দ্রতা দেখে ফসলের কাজের সময় ঠিক করুন।",
        "বপনের আগে মাটির আর্দ্রতা পরীক্ষা করুন।",
        "বড় সিদ্ধান্তের আগে স্থানীয় কৃষি পরামর্শ নিন।"
      ],

      hi: [
        "स्थानीय वर्षा और मिट्टी की नमी देखकर खेत का काम तय करें।",
        "बुवाई से पहले मिट्टी की नमी जांचें।",
        "बड़े निर्णय से पहले स्थानीय कृषि सलाह लें।"
      ],

      or: [
        "ସ୍ଥାନୀୟ ବର୍ଷା ଓ ମାଟିର ଆର୍ଦ୍ରତା ଦେଖି ଚାଷ କାମର ସମୟ ଧାର୍ଯ୍ୟ କରନ୍ତୁ।",
        "ବିଆ ବୋପିବା ପୂର୍ବରୁ ମାଟିର ଆର୍ଦ୍ରତା ଯାଞ୍ଚ କରନ୍ତୁ।",
        "ବଡ଼ ନିଷ୍ପତ୍ତି ପୂର୍ବରୁ ସ୍ଥାନୀୟ କୃଷି ପରାମର୍ଶ ନିଅନ୍ତୁ।"
      ]

    }[
      state.language
    ] || [
      "Check local rainfall and soil moisture before field decisions."
    ];


    return {
      advice,
      crop,
      actions
    };
  }


  function renderAdvisory() {

    if (!state.latest) {
      return;
    }


    const data =
      advisoryData();


    const advice =
      data.advice;


    text(
      "advisoryHeading",
      advice[0]
    );


    text(
      "advisorySeverity",
      advice[1]
    );


    const location =
      state.latest
        .gridLocation;


    text(
      "advisoryLocation",
      `${fnum(location.latitude)}°N, ` +
      `${fnum(location.longitude)}°E`
    );


    text(
      "advisoryMessage",
      `${advice[2]} Crop: ${data.crop}.`
    );


    const actions =
      $("advisoryActions");


    if (actions) {

      actions.innerHTML =
        data.actions
          .map(
            item =>
              `<li>${esc(
                item
              )}</li>`
          )
          .join("");
    }


    const farmerLocation =
      $("farmerLocation");


    if (farmerLocation) {

      farmerLocation.value =
        `${fnum(location.latitude)}°N, ` +
        `${fnum(location.longitude)}°E`;
    }
  }


  /* ============================================================
     DRIVER DISPLAY
  ============================================================ */

  function renderDrivers() {

    setWidth(
      "driverRainfall",
      75
    );


    setWidth(
      "driverDrySpell",
      pct(
        state.latest
          ?.break
          ?.[
            "7"
          ]
          ?.probability
      ) ??
      55
    );


    setWidth(
      "driverEnso",
      35
    );


    setWidth(
      "driverIod",
      30
    );


    setWidth(
      "driverMjo",
      25
    );


    text(
      "driverStatus",
      "Explanatory interface — not per-prediction feature attribution."
    );
  }


  /* ============================================================
     METRICS
  ============================================================ */

  function renderMetrics() {

    text(
      "metricOnset7Accuracy",
      `${(
        METRICS
          .onset7Accuracy *
        100
      ).toFixed(1)}%`
    );


    text(
      "metricOnset14Auc",
      METRICS
        .onset14Auc
        .toFixed(3)
    );


    text(
      "metricOnset30Auc",
      METRICS
        .onset30Auc
        .toFixed(3)
    );


    text(
      "metricBreak7Auc",
      METRICS
        .break7Auc
        .toFixed(3)
    );


    text(
      "metricBreak14Auc",
      METRICS
        .break14Auc
        .toFixed(3)
    );


    text(
      "metricBreak30Auc",
      METRICS
        .break30Auc
        .toFixed(3)
    );
  }


  /* ============================================================
     HORIZON SELECTOR
  ============================================================ */

  function setupHorizons() {

    $$(
      "[data-horizon]"
    )
      .forEach(
        button => {

          button.addEventListener(
            "click",
            () => {

              const horizon =
                num(
                  button
                    .dataset
                    .horizon
                );


              if (
                ![
                  7,
                  14,
                  30
                ].includes(
                  horizon
                )
              ) {
                return;
              }


              state.selectedHorizon =
                horizon;


              $$(
                "[data-horizon]"
              )
                .forEach(
                  item => {

                    const active =
                      num(
                        item
                          .dataset
                          .horizon
                      ) ===
                      horizon;


                    item.classList
                      .toggle(
                        "active",
                        active
                      );


                    item.setAttribute(
                      "aria-pressed",
                      String(
                        active
                      )
                    );
                  }
                );


              if (
                state.latest
              ) {

                updateLayer();

                renderAdvisory();
              }
            }
          );
        }
      );
  }


  /* ============================================================
     MAP LAYER BUTTONS
  ============================================================ */

  function setupLayers() {

    $$(
      "[data-layer]"
    )
      .forEach(
        button => {

          button.addEventListener(
            "click",
            () => {

              const layer =
                button
                  .dataset
                  .layer;


              if (
                ![
                  "grid",
                  "rainfall",
                  "onset",
                  "break"
                ].includes(
                  layer
                )
              ) {
                return;
              }


              state.activeLayer =
                layer;


              $$(
                "[data-layer]"
              )
                .forEach(
                  item => {

                    const active =
                      item
                        .dataset
                        .layer ===
                      layer;


                    item.classList
                      .toggle(
                        "active",
                        active
                      );


                    item.setAttribute(
                      "aria-pressed",
                      String(
                        active
                      )
                    );
                  }
                );


              updateLayer();
            }
          );
        }
      );
  }


  /* ============================================================
     ADVISORY CONTROLS
  ============================================================ */

  function setupAdvisory() {

    $(
      "languageSelect"
    )?.addEventListener(
      "change",
      event => {

        state.language =
          event.target
            .value ||
          "en";


        renderAdvisory();
      }
    );


    $(
      "cropSelect"
    )?.addEventListener(
      "change",
      event => {

        state.crop =
          event.target
            .value ||
          "rice";


        renderAdvisory();
      }
    );


    $(
      "generateAdvisoryButton"
    )?.addEventListener(
      "click",
      () => {

        if (
          !state.latest
        ) {

          error(
            "Run a monsoon analysis before generating an advisory."
          );

          return;
        }


        renderAdvisory();


        $(
          "advisoryMainCard"
        )?.scrollIntoView(
          {
            behavior:
              "smooth",

            block:
              "center"
          }
        );
      }
    );
  }


  /* ============================================================
     FARMER MODE
  ============================================================ */

  function setupFarmer() {

    const modal =
      $("farmerModal");


    $(
      "farmerModeButton"
    )?.addEventListener(
      "click",
      () => {

        syncFarmer();


        if (
          modal?.showModal
        ) {

          modal.showModal();

        } else {

          modal
            ?.setAttribute(
              "open",
              ""
            );
        }
      }
    );


    $(
      "closeFarmerModeButton"
    )?.addEventListener(
      "click",
      () => {

        modal?.close?.();
      }
    );


    $(
      "farmerModalGenerateButton"
    )?.addEventListener(
      "click",
      syncFarmer
    );


    modal?.addEventListener(
      "click",
      event => {

        if (
          event.target ===
          modal
        ) {

          modal.close?.();
        }
      }
    );
  }


  function syncFarmer() {

    if (
      !state.latest
    ) {

      text(
        "farmerModalLocation",
        "Run analysis first"
      );


      text(
        "farmerModalDate",
        "—"
      );


      text(
        "farmerOnsetValue",
        "—"
      );


      text(
        "farmerBreakValue",
        "—"
      );


      text(
        "farmerModalMessage",
        "Run a monsoon analysis to generate a farmer-friendly update."
      );


      html(
        "farmerModalActions",
        "<li>Awaiting model analysis</li>"
      );


      text(
        "farmerModalLanguage",
        languageName(
          state.language
        )
      );


      return;
    }


    const location =
      state.latest
        .gridLocation;


    const data =
      advisoryData();


    text(
      "farmerModalLocation",
      `${fnum(location.latitude)}°N, ` +
      `${fnum(location.longitude)}°E`
    );


    text(
      "farmerModalDate",
      fdate(
        state.latest.date
      )
    );


    text(
      "farmerOnsetValue",
      fpct(
        state.latest
          .onset
          ["7"]
          ?.probability
      )
    );


    text(
      "farmerBreakValue",
      fpct(
        state.latest
          .break
          ["7"]
          ?.probability
      )
    );


    text(
      "farmerModalMessage",
      `${data.advice[2]} ${data.crop}.`
    );


    html(
      "farmerModalActions",
      data.actions
        .map(
          item =>
            `<li>${esc(
              item
            )}</li>`
        )
        .join("")
    );


    text(
      "farmerModalLanguage",
      languageName(
        state.language
      )
    );
  }


  function languageName(
    language
  ) {

    return {

      en:
        "English",

      bn:
        "বাংলা",

      hi:
        "हिंदी",

      or:
        "ଓଡ଼ିଆ"

    }[
      language
    ] ||
    "English";
  }


  /* ============================================================
     DEMO MODE
  ============================================================ */

  function setupDemo() {

    $(
      "demoModeButton"
    )?.addEventListener(
      "click",
      () => {

        show(
          "demoSection"
        );


        $(
          "demoSection"
        )?.scrollIntoView(
          {
            behavior:
              "smooth",

            block:
              "start"
          }
        );
      }
    );


    $(
      "closeDemoButton"
    )?.addEventListener(
      "click",
      () => {

        hide(
          "demoSection"
        );
      }
    );


    $$(
      "[data-demo]"
    )
      .forEach(
        button => {

          button.addEventListener(
            "click",
            () => {

              demo(
                button
                  .dataset
                  .demo
              );
            }
          );
        }
      );
  }


  function demo(
    kind
  ) {

    const scenarios = {

      onset: [
        [
          82,
          72,
          64
        ],

        [
          18,
          26,
          34
        ]
      ],

      break: [
        [
          24,
          30,
          37
        ],

        [
          84,
          76,
          68
        ]
      ],

      mixed: [
        [
          59,
          54,
          51
        ],

        [
          61,
          58,
          52
        ]
      ]

    };


    const scenario =
      scenarios[
        kind
      ];


    if (!scenario) {
      return;
    }


    const makePrediction =
      value => ({

        probability:
          value / 100,

        threshold:
          null,

        signal:
          value >= 70
      });


    const latitude =
      num(
        $("latitude")
          ?.value
      ) ||
      DEFAULTS.lat;


    const longitude =
      num(
        $("longitude")
          ?.value
      ) ||
      DEFAULTS.lon;


    state.latest = {

      requestedLocation: {
        latitude,
        longitude
      },


      gridLocation: {
        latitude,
        longitude
      },


      date:
        $("date")
          ?.value ||
        DEFAULTS.date,


      onset: {

        7:
          makePrediction(
            scenario[0][0]
          ),

        14:
          makePrediction(
            scenario[0][1]
          ),

        30:
          makePrediction(
            scenario[0][2]
          )
      },


      break: {

        7:
          makePrediction(
            scenario[1][0]
          ),

        14:
          makePrediction(
            scenario[1][1]
          ),

        30:
          makePrediction(
            scenario[1][2]
          )
      },


      rainfall:
        null,

      climate:
        null,

      heavy:
        null,


      info: {

        prediction_type:
          "DEMO / SIMULATION",

        note:
          "Demonstration scenario — not a live prediction."
      }
    };


    renderAll();


    text(
      "modelTypeText",
      "Demo scenario"
    );


    text(
      "modelStatusText",
      `DEMO — ${kind}`
    );


    show(
      "results"
    );
  }


  /* ============================================================
     REPORT EXPORT
  ============================================================ */

  function setupReport() {

    $(
      "reportButton"
    )?.addEventListener(
      "click",
      exportReport
    );
  }


  function exportReport() {

    if (
      !state.latest
    ) {

      error(
        "Run an analysis before generating a report."
      );

      return;
    }


    const prediction =
      state.latest;


    const location =
      prediction
        .gridLocation;


    const lines = [

      "MONSOON GUARDIAN",

      "SIH 26086",

      "",

      "EXPERIMENTAL MONSOON ANALYSIS REPORT",

      "",

      `Location: ${fnum(
        location.latitude
      )}°N, ${fnum(
        location.longitude
      )}°E`,

      `Analysis date: ${fdate(
        prediction.date
      )}`,

      "",

      "MONSOON ONSET",

      `7 days: ${fpct(
        prediction
          .onset
          ["7"]
          ?.probability
      )}`,

      `14 days: ${fpct(
        prediction
          .onset
          ["14"]
          ?.probability
      )}`,

      `30 days: ${fpct(
        prediction
          .onset
          ["30"]
          ?.probability
      )}`,

      "",

      "MONSOON BREAK",

      `7 days: ${fpct(
        prediction
          .break
          ["7"]
          ?.probability
      )}`,

      `14 days: ${fpct(
        prediction
          .break
          ["14"]
          ?.probability
      )}`,

      `30 days: ${fpct(
        prediction
          .break
          ["30"]
          ?.probability
      )}`,

      "",

      "ADVISORY",

      $("advisoryMessage")
        ?.textContent ||
        "No advisory generated.",

      "",

      "Experimental development-stage output.",

      "Not an official IMD onset or break declaration."
    ];


    const blob =
      new Blob(
        [
          lines.join(
            "\n"
          )
        ],
        {
          type:
            "text/plain;charset=utf-8"
        }
      );


    const url =
      URL.createObjectURL(
        blob
      );


    const anchor =
      document.createElement(
        "a"
      );


    anchor.href =
      url;


    anchor.download =
      `monsoon-guardian-${
        prediction.date ||
        "report"
      }.txt`;


    document.body.appendChild(
      anchor
    );


    anchor.click();


    anchor.remove();


    setTimeout(
      () => {

        URL.revokeObjectURL(
          url
        );

      },
      500
    );
  }


  /* ============================================================
     NAVIGATION
  ============================================================ */

  function setupNav() {

    const nav =
      $("mainNav");


    const mobileButton =
      $("mobileMenuButton");


    mobileButton?.addEventListener(
      "click",
      () => {

        const open =
          nav?.classList.toggle(
            "open"
          );


        mobileButton.setAttribute(
          "aria-expanded",
          String(
            Boolean(
              open
            )
          )
        );
      }
    );


    $$(
      "[data-nav]"
    )
      .forEach(
        link => {

          link.addEventListener(
            "click",
            event => {

              const href =
                link.getAttribute(
                  "href"
                );


              if (
                !href ||
                !href.startsWith(
                  "#"
                )
              ) {
                return;
              }


              const target =
                document.querySelector(
                  href
                );


              if (!target) {
                return;
              }


              event.preventDefault();


              $$(
                "[data-nav]"
              )
                .forEach(
                  item =>
                    item.classList
                      .remove(
                        "active"
                      )
                );


              link.classList.add(
                "active"
              );


              target.scrollIntoView(
                {
                  behavior:
                    "smooth",

                  block:
                    "start"
                }
              );


              nav?.classList
                .remove(
                  "open"
                );


              mobileButton
                ?.setAttribute(
                  "aria-expanded",
                  "false"
                );
            }
          );
        }
      );
  }


  /* ============================================================
     RESIZE
  ============================================================ */

  function setupResize() {

    window.addEventListener(
      "resize",
      () => {

        clearTimeout(
          setupResize.timer
        );


        setupResize.timer =
          setTimeout(
            () => {

              try {

                state.map
                  ?.invalidateSize();

              } catch {
                /* ignore */
              }


              if (
                state.latest
                  ?.rainfall
              ) {

                renderRainfall(
                  state.latest
                    .rainfall
                );
              }

            },
            150
          );
      }
    );
  }


  /* ============================================================
     FORM
  ============================================================ */

  function setupForm() {

    const form =
      $("predictionForm");


    if (!form) {

      console.error(
        "Prediction form #predictionForm not found."
      );

      return;
    }


    form.addEventListener(
      "submit",
      runPrediction
    );
  }


  /* ============================================================
     INITIALIZATION
  ============================================================ */

  function init() {

    hide(
      "loading"
    );


    hide(
      "error"
    );


    hide(
      "results"
    );


    hide(
      "demoSection"
    );


    text(
      "engineStatusText",
      "Checking engine..."
    );


    text(
      "gridStatusText",
      "Grid Enabled"
    );


    text(
      "analysisStateText",
      "Analysis Ready"
    );


    text(
      "mapInstruction",
      "Click a grid point to select a location"
    );


    text(
      "modelTypeText",
      "Rainfall + Climate"
    );


    text(
      "modelStatusText",
      "Ready"
    );


    text(
      "rainfallChartStatus",
      "Awaiting analysis"
    );


    text(
      "heavyRainfallStatus",
      "EXTENSION MODULE"
    );


    text(
      "heavyRainfallModuleStatus",
      "Planned"
    );


    text(
      "advisoryHeading",
      "Awaiting prediction"
    );


    text(
      "advisorySeverity",
      "—"
    );


    text(
      "advisoryLocation",
      "Select a grid location"
    );


    text(
      "advisoryMessage",
      "Run a monsoon analysis to generate the experimental agricultural advisory."
    );


    html(
      "advisoryActions",
      "<li>Awaiting model analysis</li>"
    );


    text(
      "ensoValue",
      "—"
    );


    text(
      "iodValue",
      "—"
    );


    text(
      "mjoValue",
      "—"
    );


    renderMetrics();


    setupInputs();

    $("liveWeatherRefresh")?.addEventListener(
      "click",
      () => {
        const selected = inputs();
        loadLiveWeather(selected.lat, selected.lon);
      }
    );

    setupForm();

    setupHorizons();

    setupLayers();

    setupAdvisory();

    setupFarmer();

    setupDemo();

    setupReport();

    setupNav();

    setupResize();

    initMap();


    const initial =
      inputs();


    if (
      inside(
        initial.lat,
        initial.lon
      )
    ) {

      setMarker(
        initial.lat,
        initial.lon,
        "Default Location"
      );
    }

    loadLiveWeather(initial.lat, initial.lon);


    health();

    info();

    loadLocations();


    console.log(
      "✓ Monsoon Guardian frontend loaded",
      API
    );
  }


  /* ============================================================
     PUBLIC DEBUG API
  ============================================================ */

  window.MonsoonGuardian = {

    config: {
      API,
      BOUNDS
    },


    state,


    runPrediction,


    checkBackendHealth:
      health,


    loadGridLocations:
      loadLocations,


    initializeMap:
      initMap,


    exportReport:
      exportReport
  };


  /* ============================================================
     START
  ============================================================ */

  if (
    document.readyState ===
    "loading"
  ) {

    document.addEventListener(
      "DOMContentLoaded",
      init,
      {
        once:
          true
      }
    );

  } else {

    init();
  }

})();
