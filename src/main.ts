import * as http from "http";

import { Direction, Game } from "./game-engine";

const game = new Game(4);
game.reset();
console.log(game.toString());

http.createServer(function (req, res) {
    switch (req.url) {
        case "/reset":
            game.reset();
            res.writeHead(200, { "Content-Type": "text/json" });
            res.end(game.toJson());
            break;
        case "/action":
            let direction = "";
            req.on("data", (chunk) => {
                direction += chunk;
            });
            req.on("end", () => {
                game.turn(direction as Direction);
                res.writeHead(200, { "Content-Type": "text/json" });
                res.end(game.toJson());
            });
            break;
        case "/state":
            res.writeHead(200, { "Content-Type": "text/json" });
            res.end(game.toJson());
            break;
        default:
            res.writeHead(404);
            res.end(`'${req.url}' is not a valid path.`);
    }
}).listen(8881);
